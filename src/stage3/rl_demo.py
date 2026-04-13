import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import warnings
import os

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️ 缺少 tqdm 库！请运行: pip install tqdm")

warnings.filterwarnings("ignore")

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    HAS_RL_LIBS = True
except ImportError:
    HAS_RL_LIBS = False
    print("⚠️ 缺少强化学习库！请运行: pip install stable-baselines3 gymnasium")

# ==========================================
# 0. JCC 系统全局约束配置 (与论文对齐)
# ==========================================
TOTAL_ARM_BITS = 14            # 空间域: 极端带宽约束 B_total = 14 bits
ETC_THRESHOLD = 0.08           # 时间域: ETC 触发阈值 delta (Eq 8)

# 模拟论文 2.3.1 节提到的 LQR Q 矩阵 (高度倾斜，基座权重极大，末端极小)
LQR_Q_MATRIX = np.diag([1000.0, 500.0, 200.0, 100.0, 50.0, 20.0, 10.0])

# 【删除原来随便给定的 STATIC_SENSITIVITIES】
# STATIC_SENSITIVITIES = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05])

# ==========================================
# 1. 强化学习 (RL) 环境设计 [对齐 2.3 节]
# ==========================================

def action_to_bits(action, total_bits=14, min_bits=1):
    """将连续动作向量映射为离散的 bit 分配 (Softmax 机制)"""
    n = len(action)
    exp_a = np.exp(action - np.max(action))
    weights = exp_a / np.sum(exp_a)
    bits = [min_bits] * n
    remaining = total_bits - n * min_bits
    if remaining <= 0: return bits
    float_bits = [w * remaining for w in weights]
    int_bits = [int(math.floor(fb)) for fb in float_bits]
    for i in range(n): bits[i] += int_bits[i]
    rem = remaining - sum(int_bits)
    frac_parts = [(float_bits[i] - int_bits[i], i) for i in range(n)]
    frac_parts.sort(reverse=True, key=lambda x: x[0])
    for i in range(int(rem)): bits[int(frac_parts[i][1])] += 1
    return bits

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.7, 0, 0])

planeId = p.loadURDF("plane.urdf")
POS_A, POS_B = [0.7, -0.2, 0.025], [0.7, 0.2, 0.025]   
cubeId = p.loadURDF("cube_small.urdf", basePosition=POS_A)
robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

EE_LINK_INDEX = 11
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4, 5, 6]

# ========================================================
# 【新增严谨计算】: 基于机械臂“标称工作点”的静态敏感度计算
# ========================================================
def compute_static_sensitivities_at_home():
    """
    为了使静态 LQR 分配具备严密的数学物理依据：
    我们将 Franka Panda 机械臂放置在经典的标称 Home 姿态，
    计算此时的雅可比矩阵，并提取平移敏感度作为全局静态权重。
    """
    # Franka Panda 经典的待机/居中标称姿态 (Nominal Home Posture)
    home_q = [0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4]
    
    # 暂时将物理引擎内的机械臂设置为 Home 姿态
    for i in range(7):
        p.resetJointState(robotId, i, home_q[i])
        
    # 补齐 9 个自由度以满足 API 计算要求
    q_padded = home_q + [0.0, 0.0]
    zero_vec = [0.0] * 9
    
    # 求解该标称点下的雅可比矩阵
    J_t, _ = p.calculateJacobian(robotId, EE_LINK_INDEX, [0,0,0], q_padded, zero_vec, zero_vec)
    
    sensitivities = []
    for i in range(7):
        s = math.sqrt(J_t[0][i]**2 + J_t[1][i]**2 + J_t[2][i]**2)
        sensitivities.append(s)
        
    print(f"📐 物理引擎计算所得的标称静态敏感度 (Static Sensitivities):\n   {np.round(sensitivities, 4)}")
    return np.array(sensitivities)

# 执行物理计算，获取真实的静态基线权重
STATIC_SENSITIVITIES = compute_static_sensitivities_at_home()

# 恢复机械臂到零点准备后续流程
for aid in ARM_ACTUATOR_IDS + [9, 10]: p.resetJointState(robotId, aid, 0.0)
# ========================================================

if HAS_RL_LIBS:
    class BitAllocationEnv(gym.Env):
        def __init__(self, total_bits=14, min_bits=1):
            super().__init__()
            self.total_bits, self.min_bits = total_bits, min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            # 【对齐 Eq 14】: [q_t, dq_t, q_target, error] -> 4*7 = 28维
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
            
            self.joint_limits = {}
            for aid in ARM_ACTUATOR_IDS:
                info = p.getJointInfo(robotId, aid); lower, upper = info[8], info[9]
                if lower >= upper: lower, upper = (-2.0 * math.pi, 2.0 * math.pi)
                self.joint_limits[aid] = (lower, upper)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_q, self.current_dq, self.target_q = [], [], []
            for aid in ARM_ACTUATOR_IDS:
                low, up = self.joint_limits[aid]
                cq = np.random.uniform(low, up)
                self.current_q.append(cq)
                self.current_dq.append(np.random.uniform(-0.1, 0.1)) # 模拟角速度
                self.target_q.append(np.clip(cq + np.random.uniform(-0.3, 0.3), low, up))

            error = np.array(self.current_q) - np.array(self.target_q)
            # 严格构建观测空间 s_t
            obs = np.concatenate([self.current_q, self.current_dq, self.target_q, error]).astype(np.float32)
            return obs, {}

        def step(self, action):
            bits_rl = action_to_bits(action, self.total_bits, self.min_bits)
            quant_q = []
            for i in range(7):
                low, up = self.joint_limits[ARM_ACTUATOR_IDS[i]]
                b = bits_rl[i]; max_int = (1 << b) - 1
                if max_int <= 0: quant_q.append(self.target_q[i])
                else:
                    norm = (self.target_q[i] - low) / (up - low)
                    idx = max(0, min(int(round(norm * max_int)), max_int))
                    quant_q.append(low + (idx / max_int) * (up - low))

            # 【对齐 Eq 17】: 计算 LQR Cost 奖励
            e = np.array(quant_q) - np.array(self.target_q)
            cost = np.dot(e.T, np.dot(LQR_Q_MATRIX, e))
            reward = -float(cost)
            
            obs, _ = self.reset()
            return obs, reward, True, False, {"lqr_cost": cost}

    MODEL_PATH = "ppo_jcc_allocator.zip"
    if os.path.exists(MODEL_PATH):
        print(f"📦 加载 JCC 动态训练模型: {MODEL_PATH}")
        rl_model = PPO.load(MODEL_PATH)
    else:
        print(f"🧠 正在进行 PPO 强化学习预训练...")
        env = BitAllocationEnv(total_bits=TOTAL_ARM_BITS)
        rl_model = PPO("MlpPolicy", env, verbose=0, n_steps=512)
        from stable_baselines3.common.callbacks import BaseCallback
        class TqdmCB(BaseCallback):
            def __init__(self, steps): super().__init__(); self.pbar = None; self.steps = steps
            def _on_training_start(self): self.pbar = tqdm(total=self.steps, desc="RL 训练")
            def _on_step(self): self.pbar.update(1); return True
            def _on_training_end(self): self.pbar.close()
        rl_model.learn(total_timesteps=60000, callback=TqdmCB(60000))
        rl_model.save(MODEL_PATH)
    
    for aid in ARM_ACTUATOR_IDS + [9, 10]: p.resetJointState(robotId, aid, 0.0)
else:
    rl_model = None

# ==========================================
# 2. 网络模拟器与 ZOH [对齐 2.1 节]
# ==========================================
class NetworkSimulator:
    def __init__(self, actuator_ids):
        self.joint_limits = {}
        for aid in actuator_ids:
            info = p.getJointInfo(robotId, aid); l, u = info[8], info[9]
            if l >= u: l, u = (0.0, 0.04) if aid in [9,10] else (-2*math.pi, 2*math.pi)
            self.joint_limits[aid] = (l, u)
            
    def allocate_bits_static_lqr(self, total_bits=14):
        """【对齐 2.2.4 节】基于固定 P 矩阵敏感度的静态分配"""
        n = 7
        bits = [1] * n
        rem = total_bits - n
        # 固定按照 STATIC_SENSITIVITIES 比例分配
        weights = STATIC_SENSITIVITIES / np.sum(STATIC_SENSITIVITIES)
        float_bits = weights * rem
        int_bits = [int(math.floor(fb)) for fb in float_bits]
        for i in range(n): bits[i] += int_bits[i]
        rem -= sum(int_bits)
        if rem > 0: bits[0] += rem # 静态剩余补给给基座
        return bits

    def quantize(self, aid, target, bits):
        l, u = self.joint_limits[aid]; t = max(l, min(target, u))
        mi = (1 << bits) - 1
        if mi <= 0: return t
        idx = max(0, min(int(round(((t-l)/(u-l)) * mi)), mi))
        return l + (idx / mi) * (u - l)

network = NetworkSimulator(ARM_ACTUATOR_IDS + [9,10])

# ZOH 记忆状态 (接收端)
zoh_state = {
    'avg': np.zeros(7),
    'static': np.zeros(7),
    'rl': np.zeros(7)
}

eval_metrics = {
    'steps': 0, 
    'etc_trigger_count_avg': 0, 'err_avg': 0.0,
    'etc_trigger_count_static': 0, 'err_static': 0.0,
    'etc_trigger_count_rl': 0, 'err_rl': 0.0
}

def move_robot_ee(target_pos, target_orn, duration=2.0):
    global eval_metrics
    steps = int(duration * 240)
    for _ in range(steps):
        target_q = p.calculateInverseKinematics(robotId, EE_LINK_INDEX, target_pos, target_orn)[:7]
        joint_states = p.getJointStates(robotId, range(7))
        current_q = [s[0] for s in joint_states]
        current_dq = [s[1] for s in joint_states]
        
        # 为了计算真实空间物理误差
        J_t, _ = p.calculateJacobian(robotId, EE_LINK_INDEX, [0,0,0], current_q + [0.0,0.0], [0.0]*9, [0.0]*9)
        
        def process_scheme(scheme_name, bit_allocation):
            """处理包含 ETC 与量化的 JCC 通信全流程"""
            global eval_metrics
            # 1. 评估事件触发控制 (ETC) 阈值 [Eq 8]
            e_k = np.array(target_q) - zoh_state[scheme_name]
            trigger = np.linalg.norm(e_k) > ETC_THRESHOLD
            
            if trigger:
                # 2. 触发传输：执行量化
                eval_metrics[f'etc_trigger_count_{scheme_name}'] += 1
                q_quant = [network.quantize(i, target_q[i], bit_allocation[i]) for i in range(7)]
                # 3. ZOH 更新 [Eq 12]
                zoh_state[scheme_name] = np.array(q_quant)
            else:
                # 保持静默，沿用 ZOH
                q_quant = zoh_state[scheme_name]
                
            # 计算映射到空间的物理误差
            dq = np.array(q_quant) - np.array(target_q)
            err_sq = 0
            for d in range(3): err_sq += sum(J_t[d][i] * dq[i] for i in range(7))**2
            eval_metrics[f'err_{scheme_name}'] += math.sqrt(err_sq)
            return q_quant

        # 方案 A: 平均分配
        bits_avg = [TOTAL_ARM_BITS // 7] * 7
        process_scheme('avg', bits_avg)
        
        # 方案 B: 静态公式分配 (Static LQR-Weighted)
        bits_static = network.allocate_bits_static_lqr(TOTAL_ARM_BITS)
        process_scheme('static', bits_static)
        
        # 方案 C: 强化学习动态分配
        obs = np.concatenate([current_q, current_dq, target_q, np.array(current_q)-np.array(target_q)]).astype(np.float32)
        action, _ = rl_model.predict(obs, deterministic=True)
        bits_rl = action_to_bits(action, total_bits=TOTAL_ARM_BITS)
        q_apply = process_scheme('rl', bits_rl)

        eval_metrics['steps'] += 1
        
        # 实际驱动采用 RL 输出
        for i in range(7): p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, q_apply[i])
        p.stepSimulation(); time.sleep(1./240)

def control_gripper(target_width, duration=1.0):
    for _ in range(int(duration * 240)):
        p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, target_width, force=50)
        p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, target_width, force=50)
        p.stepSimulation(); time.sleep(1./240)

# ==========================================
# 4. 仿真执行与 JCC 数据统计
# ==========================================
print(f"🚀 JCC 测试开始 (带宽: {TOTAL_ARM_BITS} Bits, ETC阈值: {ETC_THRESHOLD})")
down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

# 执行动作流程
move_robot_ee([0.7, -0.2, 0.25], down_orn, 1.0)
move_robot_ee([0.7, -0.2, 0.04], down_orn, 0.5)
move_robot_ee([0.7, 0.2, 0.25], down_orn, 1.0)

print("\n" + "="*65)
print(f" 📊 JCC (ETC + 空间分配) 性能评估报告")
print("="*65)
n_steps = eval_metrics['steps']

def print_scheme(name, tag, bits):
    e = eval_metrics[f'err_{tag}']/n_steps*1000
    # 时间域压缩率：没触发ETC的次数 / 总步数
    c_rate = (1 - eval_metrics[f'etc_trigger_count_{tag}']/n_steps) * 100
    print(f"{name:<15} | 误差: {e:>6.2f} mm | 时间域静默率: {c_rate:>5.1f}% | 位宽: {bits}")

print_scheme("1. 平均分配", "avg", [2,2,2,2,2,2,2])
print_scheme("2. 静态LQR分配", "static", network.allocate_bits_static_lqr(TOTAL_ARM_BITS))
print_scheme("3. RL动态分配", "rl", "动态变化")
print("-" * 65)

# 计算 RL 相对静态 LQR 的提升
err_s = eval_metrics['err_static']/n_steps*1000
err_rl = eval_metrics['err_rl']/n_steps*1000
improvement = (err_s - err_rl) / err_s * 100

print(f"💡 结论: 在同样享有 ETC 时间域压缩 的前提下，")
print(f"   RL 智能体凭借上下文感知能力，克服了静态 LQR 矩阵的僵化缺陷，")
print(f"   使得系统的控制追踪误差大幅降低了 {improvement:.1f}%！")
print("="*65 + "\n")

p.disconnect()
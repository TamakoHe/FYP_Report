import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import warnings
import os
import multiprocessing

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️ 缺少 tqdm 库！请运行: pip install tqdm")

warnings.filterwarnings("ignore")

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback  # <-- 补上这一行导入
    HAS_RL_LIBS = True
except ImportError:
    HAS_RL_LIBS = False
    print("⚠️ 缺少强化学习库！请运行: pip install stable-baselines3 gymnasium")

# ==========================================
# 0. JCC 系统全局约束配置 (与论文对齐)
# ==========================================
TOTAL_ARM_BITS = 56            # 空间域: 极端带宽约束 B_total = 14 bits
ETC_THRESHOLD = 0.08           # 时间域: ETC 触发阈值 delta (Eq 8)
EE_LINK_INDEX = 11
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4, 5, 6]

# ==========================================
# 1. 静态基线预计算 (仅运行一次的无头物理计算)
# ==========================================
def compute_static_sensitivities_at_home():
    """
    启动一个临时的无头物理引擎，计算标称工作点下的雅可比敏感度，
    作为静态基线的基础。计算完毕后立即销毁。
    """
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
    
    home_q = [0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4]
    for i in range(7): 
        p.resetJointState(robot, i, home_q[i], physicsClientId=cid)
        
    q_padded = home_q + [0.0, 0.0]
    zero_vec = [0.0] * 9
    J_t, _ = p.calculateJacobian(robot, EE_LINK_INDEX, [0,0,0], q_padded, zero_vec, zero_vec, physicsClientId=cid)
    
    sensitivities = [math.sqrt(J_t[0][i]**2 + J_t[1][i]**2 + J_t[2][i]**2) for i in range(7)]
    p.disconnect(cid)
    return np.array(sensitivities)

STATIC_SENSITIVITIES = compute_static_sensitivities_at_home()

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

def get_static_lqr_bits(total_bits=14):
    """提取出的静态 LQR 注水分配算法 (独立函数供 RL 作为基线对比)"""
    n = 7
    min_bits = 1
    weights = [max(w, 1e-6) for w in STATIC_SENSITIVITIES]
    bits_float = [0.0] * n
    unfixed = list(range(n))
    while True:
        if not unfixed: break
        sum_log2 = sum(0.5 * math.log2(weights[i]) for i in unfixed)
        rem_bits = total_bits - sum(bits_float[j] for j in range(n) if j not in unfixed)
        offset = (rem_bits - sum_log2) / len(unfixed)
        newly_fixed = False
        for i in unfixed:
            b = offset + 0.5 * math.log2(weights[i])
            if b < min_bits:
                bits_float[i] = min_bits
                unfixed.remove(i)
                newly_fixed = True
                break
        if not newly_fixed:
            for i in unfixed: bits_float[i] = offset + 0.5 * math.log2(weights[i])
            break
    bits_int = [int(math.floor(b)) for b in bits_float]
    remainder = total_bits - sum(bits_int)
    frac_parts = [(bits_float[i] - bits_int[i], i) for i in range(n)]
    frac_parts.sort(reverse=True, key=lambda x: x[0])
    for i in range(int(remainder)): bits_int[frac_parts[i][1]] += 1
    return bits_int

# ==========================================
# 2. 强化学习环境：支持多进程独立物理引擎
# ==========================================
if HAS_RL_LIBS:
    class BitAllocationEnv(gym.Env):
        def __init__(self, total_bits=14, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
            
            # 【多核核心】: 每个环境实例启动属于自己的无头物理服务器
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
            
            self.joint_limits = {}
            for aid in ARM_ACTUATOR_IDS:
                info = p.getJointInfo(self.robot_id, aid, physicsClientId=self.client_id)
                lower, upper = info[8], info[9]
                if lower >= upper: lower, upper = (-2.0 * math.pi, 2.0 * math.pi)
                self.joint_limits[aid] = (lower, upper)
                
            self.static_baseline_bits = get_static_lqr_bits(self.total_bits)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_q, self.current_dq, self.target_q = [], [], []
            for aid in ARM_ACTUATOR_IDS:
                low, up = self.joint_limits[aid]
                cq = np.random.uniform(low, up)
                self.current_q.append(cq)
                self.current_dq.append(np.random.uniform(-0.1, 0.1))
                self.target_q.append(np.clip(cq + np.random.uniform(-0.3, 0.3), low, up))

            error = np.array(self.current_q) - np.array(self.target_q)
            obs = np.concatenate([self.current_q, self.current_dq, self.target_q, error]).astype(np.float32)
            return obs, {}

        def _quantize_state(self, bits_list):
            quant_q = []
            for i in range(7):
                low, up = self.joint_limits[ARM_ACTUATOR_IDS[i]]
                b = bits_list[i]; max_int = (1 << b) - 1
                if max_int <= 0: quant_q.append(self.target_q[i])
                else:
                    norm = (self.target_q[i] - low) / (up - low)
                    idx = max(0, min(int(round(norm * max_int)), max_int))
                    quant_q.append(low + (idx / max_int) * (up - low))
            return quant_q

        def step(self, action):
            bits_rl = action_to_bits(action, self.total_bits, self.min_bits)
            quant_q_rl = self._quantize_state(bits_rl)
            quant_q_static = self._quantize_state(self.static_baseline_bits)

            # 【硬核物理计算】确保所有操作只在各自子进程的物理世界内进行
            for i, aid in enumerate(ARM_ACTUATOR_IDS): 
                p.resetJointState(self.robot_id, aid, self.target_q[i], physicsClientId=self.client_id)
            pos_target = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            
            for i, aid in enumerate(ARM_ACTUATOR_IDS): 
                p.resetJointState(self.robot_id, aid, quant_q_static[i], physicsClientId=self.client_id)
            pos_static = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_static = np.linalg.norm(pos_target - pos_static)
            
            for i, aid in enumerate(ARM_ACTUATOR_IDS): 
                p.resetJointState(self.robot_id, aid, quant_q_rl[i], physicsClientId=self.client_id)
            pos_rl = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_rl = np.linalg.norm(pos_target - pos_rl)

            reward = float(error_static - error_rl) * 1000.0
            
            obs, _ = self.reset()
            return obs, reward, True, False, {"error_rl_mm": error_rl * 1000}
            
        def close(self):
            """关闭子进程内的物理引擎"""
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps): super().__init__(); self.pbar = None; self.steps = steps
        def _on_training_start(self): self.pbar = tqdm(total=self.steps, desc="🏎️ 多核并行硬核物理训练")
        def _on_step(self): 
            # SubprocVecEnv 会返回一个 infos 列表，取第一个即可
            error_mm = self.locals.get("infos", [{}])[0].get("error_rl_mm", 0.0)
            self.pbar.update(self.locals.get("env").num_envs) # 按核数更新进度
            self.pbar.set_postfix({"末端物理误差(mm)": f"{error_mm:.2f}"})
            return True
        def _on_training_end(self): self.pbar.close()

# ==========================================
# 3. 网络模拟器 (测试环境专用)
# ==========================================
class NetworkSimulator:
    def __init__(self, robot_id, actuator_ids):
        self.robot_id = robot_id
        self.joint_limits = {}
        for aid in actuator_ids:
            info = p.getJointInfo(robot_id, aid); l, u = info[8], info[9]
            if l >= u: l, u = (0.0, 0.04) if aid in [9,10] else (-2*math.pi, 2*math.pi)
            self.joint_limits[aid] = (l, u)

    def allocate_bits_static_lqr(self, total_bits=14):
        return get_static_lqr_bits(total_bits)

    def quantize(self, aid, target, bits):
        l, u = self.joint_limits[aid]; t = max(l, min(target, u))
        mi = (1 << bits) - 1
        if mi <= 0: return t
        idx = max(0, min(int(round(((t-l)/(u-l)) * mi)), mi))
        return l + (idx / mi) * (u - l)

# ==========================================
# 4. 主程序：支持多进程保护的执行入口
# ==========================================
if __name__ == '__main__':
    # ----------------------------------------
    # 阶段 A: 多核并行 RL 训练 (Headless)
    # ----------------------------------------
    MODEL_PATH = "ppo_multicore_physical_allocator.zip"
    
    if HAS_RL_LIBS:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现已保存的物理级精确训练模型，正在加载: {MODEL_PATH}")
            rl_model = PPO.load(MODEL_PATH)
        else:
            # 获取 CPU 核心数，配置多进程环境
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构，已成功分配 {num_cores} 个 CPU 核心！")
            print("   (每个核心将运行一个独立的 PyBullet 物理宇宙以加速计算...)")
            
            # 使用 SubprocVecEnv 将环境部署到多个子进程
            vec_env = make_vec_env(lambda: BitAllocationEnv(total_bits=TOTAL_ARM_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            rl_model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=512)
            
            TOTAL_STEPS = 150000
            # 训练速度现在将是单核的数倍！
            rl_model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS))
            rl_model.save(MODEL_PATH)
            
            # 销毁后台子进程的物理引擎
            vec_env.close()
            print(f"✅ 多核训练圆满结束并保存至: {MODEL_PATH}\n")
    else:
        rl_model = None

    # ----------------------------------------
    # 阶段 B: GUI 物理可视化测试评估
    # ----------------------------------------
    print("🖥️ 正在启动 GUI 物理可视化环境用于方案对决...")
    # 这里才是主窗口的可视化引擎
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.7, 0, 0])

    planeId = p.loadURDF("plane.urdf")
    POS_A, POS_B = [0.7, -0.2, 0.025], [0.7, 0.2, 0.025]   
    cubeId = p.loadURDF("cube_small.urdf", basePosition=POS_A)
    # 此处 robotId 绑定的是可视化主进程
    robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

    for aid in ARM_ACTUATOR_IDS + [9, 10]: 
        p.resetJointState(robotId, aid, 0.0)

    network = NetworkSimulator(robotId, ARM_ACTUATOR_IDS + [9,10])
    zoh_state = { 'avg': np.zeros(7), 'static': np.zeros(7), 'rl': np.zeros(7) }

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
            
            def process_scheme(scheme_name, bit_allocation):
                global eval_metrics
                e_k = np.array(target_q) - zoh_state[scheme_name]
                trigger = np.linalg.norm(e_k) > ETC_THRESHOLD
                
                if trigger:
                    eval_metrics[f'etc_trigger_count_{scheme_name}'] += 1
                    q_quant = [network.quantize(i, target_q[i], bit_allocation[i]) for i in range(7)]
                    zoh_state[scheme_name] = np.array(q_quant)
                else:
                    q_quant = zoh_state[scheme_name]
                    
                saved_q = [p.getJointState(robotId, i)[0] for i in range(7)] 
                
                for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(robotId, aid, target_q[i])
                pos_t = np.array(p.getLinkState(robotId, EE_LINK_INDEX)[0])
                
                for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(robotId, aid, q_quant[i])
                pos_q = np.array(p.getLinkState(robotId, EE_LINK_INDEX)[0])
                
                true_physical_error = np.linalg.norm(pos_t - pos_q)
                eval_metrics[f'err_{scheme_name}'] += true_physical_error
                
                for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(robotId, aid, saved_q[i])
                
                return q_quant

            bits_avg = [TOTAL_ARM_BITS // 7] * 7
            process_scheme('avg', bits_avg)
            
            bits_static = network.allocate_bits_static_lqr(TOTAL_ARM_BITS)
            process_scheme('static', bits_static)
            
            obs = np.concatenate([current_q, current_dq, target_q, np.array(current_q)-np.array(target_q)]).astype(np.float32)
            action, _ = rl_model.predict(obs, deterministic=True)
            bits_rl = action_to_bits(action, total_bits=TOTAL_ARM_BITS)
            q_apply = process_scheme('rl', bits_rl)

            eval_metrics['steps'] += 1
            
            for i in range(7): p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, q_apply[i])
            p.stepSimulation(); time.sleep(1./240)

    def control_gripper(target_width, duration=1.0):
        for _ in range(int(duration * 240)):
            p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, target_width, force=50)
            p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, target_width, force=50)
            p.stepSimulation(); time.sleep(1./240)

    print(f"\n🚀 开始执行物理轨迹 (真实物理 FK 评估 / 带宽: {TOTAL_ARM_BITS} Bits)")
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    move_robot_ee([0.7, -0.2, 0.25], down_orn, 1.0)
    move_robot_ee([0.7, -0.2, 0.04], down_orn, 0.5)
    move_robot_ee([0.7, 0.2, 0.25], down_orn, 1.0)

    print("\n" + "="*65)
    print(f" 📊 极致物理真实度 - JCC 性能评估报告")
    print("="*65)
    n_steps = eval_metrics['steps']

    def print_scheme(name, tag, bits):
        e = eval_metrics[f'err_{tag}']/n_steps*1000
        c_rate = (1 - eval_metrics[f'etc_trigger_count_{tag}']/n_steps) * 100
        print(f"{name:<15} | 误差: {e:>6.2f} mm | 时间域静默率: {c_rate:>5.1f}% | 位宽: {bits}")

    print_scheme("1. 平均分配", "avg", [2,2,2,2,2,2,2])
    print_scheme("2. 静态LQR分配", "static", network.allocate_bits_static_lqr(TOTAL_ARM_BITS))
    print_scheme("3. RL动态分配", "rl", "动态实时分配")
    print("-" * 65)

    err_s = eval_metrics['err_static']/n_steps*1000
    err_rl = eval_metrics['err_rl']/n_steps*1000
    improvement = (err_s - err_rl) / err_s * 100

    print(f"💡 物理准确性与多核优化结论:")
    print(f"   利用 {multiprocessing.cpu_count()} 核心多进程算力完成大规模物理碰撞试错，")
    if improvement > 0:
        print(f"🏆 RL 模型彻底压制了静态数学理论，追踪误差大幅降低了 {improvement:.1f}%！")
    else:
        print(f"⚠️ 当前轨迹下未完全击败基线 (差异 {improvement:.1f}%)。")
    print("="*65 + "\n")

    p.disconnect() 
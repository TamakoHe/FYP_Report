import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import warnings
import os
import multiprocessing
from typing import Callable

# 尝试导入绘图库
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ 缺少 matplotlib 库，将无法绘制 3D 轨迹图！请运行: pip install matplotlib")

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
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_RL_LIBS = True
except ImportError:
    HAS_RL_LIBS = False
    print("⚠️ 缺少强化学习库！请运行: pip install stable-baselines3 gymnasium")

# ==========================================
# 0. JCC 系统全局约束配置 (与论文对齐)
# ==========================================
TOTAL_ARM_BITS = 56            # 极端带宽约束 B_total
ETC_THRESHOLD = 0.08           # 时间域: ETC 触发阈值 delta
EE_LINK_INDEX = 11
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4, 5, 6]

# ==========================================
# 【稳定性组件】: 学习率线性衰减调度器
# ==========================================
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    线性衰减调度器。
    随着训练进度 (progress_remaining 从 1.0 降到 0.0)，逐步降低学习率和截断范围。
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 1. 静态基线预计算 (仅运行一次的无头物理计算)
# ==========================================
def compute_static_sensitivities_at_home():
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

            improvement_mm = float(error_static - error_rl) * 1000.0
            
            # 【奖励截断】 抵御异常物理姿态产生的爆炸梯度
            reward = np.clip(improvement_mm, -15.0, 15.0)
            
            obs, _ = self.reset()
            return obs, reward, True, False, {"error_rl_mm": error_rl * 1000}
            
        def close(self):
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps, eval_freq=5000): 
            super().__init__()
            self.pbar = None
            self.steps = steps
            self.eval_freq = eval_freq
            self.last_eval_step = 0
            self.eval_history = [] 

        def _on_training_start(self): 
            self.pbar = tqdm(total=self.steps, desc="🏎️ 多核稳定物理训练 & 周期验证")

        def _on_step(self): 
            self.pbar.update(self.locals.get("env").num_envs)
            
            if self.num_timesteps - self.last_eval_step >= self.eval_freq:
                self.last_eval_step = self.num_timesteps
                metrics = run_evaluation_trajectory(self.model, TOTAL_ARM_BITS, gui=False)
                err_rl = (metrics['err_rl'] / metrics['steps']) * 1000
                self.eval_history.append((self.num_timesteps, err_rl))
                self.pbar.set_postfix({"标准轨迹误差(mm)": f"{err_rl:.2f}"})
                
            return True
            
        def _on_training_end(self): 
            self.pbar.close()
            if self.eval_history:
                file_name = "eval_trajectory_errors.csv"
                np.savetxt(file_name, self.eval_history, delimiter=",", header="Step,Error_RL_mm", comments="")
                print(f"\n💾 固定标准轨迹周期验证数据已保存至 {file_name} (可完美绘制收敛曲线 Fig 1)")

# ==========================================
# 3. 统一轨迹验证核心逻辑 (用于训练周期评估与最终测试)
# ==========================================
def run_evaluation_trajectory(rl_model, total_bits, gui=False):
    cid = p.connect(p.GUI) if gui else p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.7, 0, 0], physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)
    POS_A, POS_B = [0.7, -0.2, 0.025], [0.7, 0.2, 0.025]   
    p.loadURDF("cube_small.urdf", basePosition=POS_A, physicsClientId=cid)
    robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)

    for aid in ARM_ACTUATOR_IDS + [9, 10]: 
        p.resetJointState(robotId, aid, 0.0, physicsClientId=cid)

    class LocalNetworkSimulator:
        def __init__(self, r_id, actuator_ids):
            self.joint_limits = {}
            for aid in actuator_ids:
                info = p.getJointInfo(r_id, aid, physicsClientId=cid)
                l, u = info[8], info[9]
                if l >= u: l, u = (0.0, 0.04) if aid in [9,10] else (-2*math.pi, 2*math.pi)
                self.joint_limits[aid] = (l, u)

        def allocate_bits_static_lqr(self, t_bits=14):
            return get_static_lqr_bits(t_bits)

        def quantize(self, aid, target, bits):
            l, u = self.joint_limits[aid]; t = max(l, min(target, u))
            mi = (1 << bits) - 1
            if mi <= 0: return t
            idx = max(0, min(int(round(((t-l)/(u-l)) * mi)), mi))
            return l + (idx / mi) * (u - l)

    network = LocalNetworkSimulator(robotId, ARM_ACTUATOR_IDS + [9,10])
    zoh_state = { 'avg': np.zeros(7), 'static': np.zeros(7), 'rl': np.zeros(7) }

    # 新增 trajectory_data 用于记录末端的 3D 轨迹坐标点
    eval_metrics = {
        'steps': 0, 
        'etc_trigger_count_avg': 0, 'err_avg': 0.0,
        'etc_trigger_count_static': 0, 'err_static': 0.0,
        'etc_trigger_count_rl': 0, 'err_rl': 0.0,
        'trajectory_data': {'target': [], 'avg': [], 'static': [], 'rl': []}
    }

    def move_robot_ee(target_pos, target_orn, duration=2.0):
        steps = int(duration * 240)
        for _ in range(steps):
            target_q = p.calculateInverseKinematics(robotId, EE_LINK_INDEX, target_pos, target_orn, physicsClientId=cid)[:7]
            joint_states = p.getJointStates(robotId, range(7), physicsClientId=cid)
            current_q = [s[0] for s in joint_states]
            current_dq = [s[1] for s in joint_states]
            
            def process_scheme(scheme_name, bit_allocation):
                e_k = np.array(target_q) - zoh_state[scheme_name]
                trigger = np.linalg.norm(e_k) > ETC_THRESHOLD
                
                if trigger:
                    eval_metrics[f'etc_trigger_count_{scheme_name}'] += 1
                    q_quant = [network.quantize(i, target_q[i], bit_allocation[i]) for i in range(7)]
                    zoh_state[scheme_name] = np.array(q_quant)
                else:
                    q_quant = zoh_state[scheme_name]
                    
                saved_q = [p.getJointState(robotId, i, physicsClientId=cid)[0] for i in range(7)] 
                
                # 计算理想 target 坐标
                for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(robotId, aid, target_q[i], physicsClientId=cid)
                pos_t = np.array(p.getLinkState(robotId, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=cid)[0])
                
                # 计算量化后的真实执行坐标
                for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(robotId, aid, q_quant[i], physicsClientId=cid)
                pos_q = np.array(p.getLinkState(robotId, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=cid)[0])
                
                # 保存 3D 轨迹点数据供末尾画图使用
                eval_metrics['trajectory_data'][scheme_name].append(pos_q)
                if scheme_name == 'avg':
                    eval_metrics['trajectory_data']['target'].append(pos_t)
                
                true_physical_error = np.linalg.norm(pos_t - pos_q)
                eval_metrics[f'err_{scheme_name}'] += true_physical_error
                
                # 恢复物理状态
                for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(robotId, aid, saved_q[i], physicsClientId=cid)
                
                return q_quant

            bits_avg = [total_bits // 7] * 7
            process_scheme('avg', bits_avg)
            
            bits_static = network.allocate_bits_static_lqr(total_bits)
            process_scheme('static', bits_static)
            
            obs = np.concatenate([current_q, current_dq, target_q, np.array(current_q)-np.array(target_q)]).astype(np.float32)
            if rl_model is not None:
                action, _ = rl_model.predict(obs, deterministic=True)
                bits_rl = action_to_bits(action, total_bits=total_bits)
            else:
                bits_rl = bits_static
            q_apply = process_scheme('rl', bits_rl)

            eval_metrics['steps'] += 1
            
            for i in range(7): p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, q_apply[i], physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
            if gui: time.sleep(1./240)

    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    move_robot_ee([0.7, -0.2, 0.25], down_orn, 1.0)
    move_robot_ee([0.7, -0.2, 0.04], down_orn, 0.5)
    move_robot_ee([0.7, 0.2, 0.25], down_orn, 1.0)
    
    p.disconnect(cid)
    return eval_metrics

# ==========================================
# 4. 主程序：支持多进程保护的执行入口
# ==========================================
if __name__ == '__main__':
    MODEL_PATH = "ppo_multicore_physical_allocator.zip"
    
    if HAS_RL_LIBS:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现已保存的物理级精确训练模型，正在加载: {MODEL_PATH}")
            rl_model = PPO.load(MODEL_PATH)
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构，已成功分配 {num_cores} 个 CPU 核心！")
            
            vec_env = make_vec_env(lambda: BitAllocationEnv(total_bits=TOTAL_ARM_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            rl_model = PPO("MlpPolicy", vec_env, verbose=0, n_steps=512,
                           learning_rate=linear_schedule(3e-4),
                           clip_range=linear_schedule(0.2))
            
            TOTAL_STEPS = 100000  # 修改为 10w 步
            rl_model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS, eval_freq=5000))
            rl_model.save(MODEL_PATH)
            
            vec_env.close()
            print(f"✅ 多核训练圆满结束并保存至: {MODEL_PATH}\n")
    else:
        rl_model = None

    print("🖥️ 正在启动 GUI 物理可视化环境用于最终方案对决...")
    print(f"🚀 开始执行物理轨迹 (真实物理 FK 评估 / 带宽: {TOTAL_ARM_BITS} Bits)")
    
    final_metrics = run_evaluation_trajectory(rl_model, TOTAL_ARM_BITS, gui=True)
    
    print("\n" + "="*65)
    print(f" 📊 极致物理真实度 - JCC 性能评估报告")
    print("="*65)
    n_steps = final_metrics['steps']

    def print_scheme(name, tag, bits):
        e = final_metrics[f'err_{tag}']/n_steps*1000
        c_rate = (1 - final_metrics[f'etc_trigger_count_{tag}']/n_steps) * 100
        print(f"{name:<15} | 误差: {e:>6.2f} mm | 时间域静默率: {c_rate:>5.1f}% | 位宽: {bits}")

    print_scheme("1. 平均分配", "avg", [TOTAL_ARM_BITS//7]*7)
    print_scheme("2. 静态LQR分配", "static", get_static_lqr_bits(TOTAL_ARM_BITS))
    print_scheme("3. RL动态分配", "rl", "动态实时分配")
    print("-" * 65)

    err_s = final_metrics['err_static']/n_steps*1000
    err_rl = final_metrics['err_rl']/n_steps*1000
    improvement = (err_s - err_rl) / err_s * 100

    print(f"💡 物理准确性与多核优化结论:")
    print(f"   利用 {multiprocessing.cpu_count() if HAS_RL_LIBS else 1} 核心多进程算力完成大规模物理碰撞试错，")
    if improvement > 0:
        print(f"🏆 RL 模型彻底压制了静态数学理论，追踪误差大幅降低了 {improvement:.1f}%！")
    else:
        print(f"⚠️ 当前轨迹下未完全击败基线 (差异 {improvement:.1f}%)。")
    print("="*65 + "\n")

    # ==========================================
    # 5. 绘制 3D 轨迹对比图 (Fig 2)
    # ==========================================
    if HAS_MATPLOTLIB:
        print("📈 正在生成并保存 3D 轨迹对比图 (Fig 2)...")
        traj = final_metrics['trajectory_data']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取记录的轨迹数据
        target_pts = np.array(traj['target'])
        avg_pts = np.array(traj['avg'])
        static_pts = np.array(traj['static'])
        rl_pts = np.array(traj['rl'])
        
        # 绘制三维曲线
        ax.plot(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], color='black', linestyle='--', linewidth=2.5, label='Target (Ideal)')
        ax.plot(avg_pts[:, 0], avg_pts[:, 1], avg_pts[:, 2], color='red', linestyle='-', alpha=0.6, label='Average')
        ax.plot(static_pts[:, 0], static_pts[:, 1], static_pts[:, 2], color='blue', linestyle='-', alpha=0.6, label='Static LQR')
        ax.plot(rl_pts[:, 0], rl_pts[:, 1], rl_pts[:, 2], color='green', linestyle='-', linewidth=2.5, label='RL (Proposed)')
        
        ax.set_xlabel('X Position (m)', labelpad=10)
        ax.set_ylabel('Y Position (m)', labelpad=10)
        ax.set_zlabel('Z Position (m)', labelpad=10)
        ax.set_title(f'3D End-Effector Trajectory ({TOTAL_ARM_BITS}-bit Bandwidth Constraint)', fontsize=14, pad=15)
        ax.legend(loc='upper right', fontsize=10)
        
        # 设置优雅的观测视角
        ax.view_init(elev=25, azim=45)
        
        plot_filename = "Figure_2_3D_Trajectory.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"💾 3D 轨迹图已成功保存为 {plot_filename} ！")
        plt.show()
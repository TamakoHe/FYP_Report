import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import warnings
import os
import multiprocessing
from typing import Callable

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
TOTAL_ARM_BITS = 28            
ETC_THRESHOLD = 0.0           
EE_LINK_INDEX = 11
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4, 5, 6]

COMM_FREQ_STEPS = 24
MAX_DELTA = 0.2

HOME_Q = [0.0, -math.pi/4, 0.0, -3*math.pi/4, 0.0, math.pi/2, math.pi/4]

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 1. 静态基线预计算
# ==========================================
def compute_static_sensitivities_at_home():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
    
    for i in range(7): p.resetJointState(robot, i, HOME_Q[i], physicsClientId=cid)
        
    q_padded = HOME_Q + [0.0, 0.0]
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
    n = 7; min_bits = 1
    weights = [max(w, 1e-6) for w in STATIC_SENSITIVITIES]
    bits_float = [0.0] * n; unfixed = list(range(n))
    while True:
        if not unfixed: break
        sum_log2 = sum(0.5 * math.log2(weights[i]) for i in unfixed)
        rem_bits = total_bits - sum(bits_float[j] for j in range(n) if j not in unfixed)
        offset = (rem_bits - sum_log2) / len(unfixed)
        newly_fixed = False
        for i in unfixed:
            b = offset + 0.5 * math.log2(weights[i])
            if b < min_bits:
                bits_float[i] = min_bits; unfixed.remove(i); newly_fixed = True; break
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
# 2. 强化学习环境 (带3D空间感知注入)
# ==========================================
if HAS_RL_LIBS:
    class BitAllocationEnv(gym.Env):
        def __init__(self, total_bits=14, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
            
            # 【核心升级】：状态空间从 28 维扩展为 34 维，注入 3D 空间感知！
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(34,), dtype=np.float32)
            
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
            self.avg_baseline_bits = [self.total_bits // 7] * 7

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_q, self.current_dq, self.target_q = [], [], []
            for i, aid in enumerate(ARM_ACTUATOR_IDS):
                low, up = self.joint_limits[aid]
                base_q = HOME_Q[i]
                cq = np.random.uniform(max(low, base_q - 1.5), min(up, base_q + 1.5))
                self.current_q.append(cq)
                self.current_dq.append(np.random.uniform(-0.1, 0.1))
                self.target_q.append(np.clip(cq + np.random.uniform(-MAX_DELTA, MAX_DELTA), low, up))

            # 提取当前的 3D 末端坐标
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, self.current_q[i], physicsClientId=self.client_id)
            current_ee = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            
            # 提取目标的 3D 末端坐标
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, self.target_q[i], physicsClientId=self.client_id)
            target_ee = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])

            error = np.array(self.current_q) - np.array(self.target_q)
            
            # 【空间感知拼接】：将 current_ee (3维) 和 target_ee (3维) 拼入大脑
            obs = np.concatenate([self.current_q, self.current_dq, self.target_q, error, current_ee, target_ee]).astype(np.float32)
            return obs, {}

        def _quantize_delta_state(self, bits_list):
            quant_q = []
            for i in range(7):
                delta = self.target_q[i] - self.current_q[i]
                l, u = -MAX_DELTA, MAX_DELTA
                d_clip = max(l, min(delta, u))
                mi = (1 << int(bits_list[i])) - 1
                if mi <= 0:
                    quant_q.append(self.current_q[i])
                else:
                    idx = max(0, min(int(round(((d_clip-l)/(u-l)) * mi)), mi))
                    d_quant = l + (idx / mi) * (u - l)
                    phys_l, phys_u = self.joint_limits[ARM_ACTUATOR_IDS[i]]
                    final_q = self.current_q[i] + d_quant
                    quant_q.append(max(phys_l, min(final_q, phys_u)))
            return quant_q

        def step(self, action):
            bits_rl = action_to_bits(action, self.total_bits, self.min_bits)
            
            quant_q_rl = self._quantize_delta_state(bits_rl)
            quant_q_static = self._quantize_delta_state(self.static_baseline_bits)
            quant_q_avg = self._quantize_delta_state(self.avg_baseline_bits)

            # 目标位置
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, self.target_q[i], physicsClientId=self.client_id)
            pos_target = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            
            # 1. 静态 LQR 误差
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, quant_q_static[i], physicsClientId=self.client_id)
            pos_static = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_static = np.linalg.norm(pos_target - pos_static)
            
            # 2. 均匀分配 误差
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, quant_q_avg[i], physicsClientId=self.client_id)
            pos_avg = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_avg = np.linalg.norm(pos_target - pos_avg)

            # 3. RL 误差
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, quant_q_rl[i], physicsClientId=self.client_id)
            pos_rl = np.array(p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_rl = np.linalg.norm(pos_target - pos_rl)

            # 【核心修复】：平滑融合奖励，不使用非凸的 min() 函数
            # 首要目标是打爆 Static (权重1.0)，次要目标是碾压 Avg (权重0.5)
            improvement_static = float(error_static - error_rl) * 1000.0
            improvement_avg = float(error_avg - error_rl) * 1000.0
            
            reward = np.clip(improvement_static + 0.5 * improvement_avg, -50.0, 50.0)
            
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
                err_rl = metrics['rl']['grasp'] * 1000
                self.eval_history.append((self.num_timesteps, err_rl))
                self.pbar.set_postfix({"标定轨迹Grasp稳态误差(mm)": f"{err_rl:.2f}"})
            return True
            
        def _on_training_end(self): 
            self.pbar.close()
            if self.eval_history:
                file_name = "eval_trajectory_errors.csv"
                np.savetxt(file_name, self.eval_history, delimiter=",", header="Step,Error_RL_mm", comments="")

# ==========================================
# 3. 终极验证：加入稳态测试消除动态追踪延迟！
# ==========================================
def run_evaluation_trajectory(rl_model, total_bits, gui=False):
    cid = p.connect(p.GUI) if gui else p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    if gui:
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.7, 0, 0], physicsClientId=cid)

    p.loadURDF("plane.urdf", physicsClientId=cid)
    robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)

    class LocalNetworkSimulator:
        def __init__(self, r_id, actuator_ids):
            self.joint_limits = {}
            for aid in actuator_ids:
                info = p.getJointInfo(r_id, aid, physicsClientId=cid)
                l, u = info[8], info[9]
                if l >= u: l, u = (-2*math.pi, 2*math.pi)
                self.joint_limits[aid] = (l, u)

        def quantize_delta(self, aid, target, current, bits):
            delta = target - current
            l, u = -MAX_DELTA, MAX_DELTA
            d_clip = max(l, min(delta, u))
            mi = (1 << int(bits)) - 1
            if mi <= 0: return current
            idx = max(0, min(int(round(((d_clip-l)/(u-l)) * mi)), mi))
            d_quant = l + (idx / mi) * (u - l)
            
            phys_l, phys_u = self.joint_limits[aid]
            final_q = current + d_quant
            return max(phys_l, min(final_q, phys_u))

    network = LocalNetworkSimulator(robotId, ARM_ACTUATOR_IDS)
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    
    trajectories = [
        {"hover": [0.7, -0.2, 0.25], "grasp": [0.7, -0.2, 0.04], "place": [0.7, 0.2, 0.04]},  
        {"hover": [0.6, 0.0, 0.25], "grasp": [0.6, 0.0, 0.04], "place": [0.5, -0.3, 0.04]},   
        {"hover": [0.5, 0.3, 0.25], "grasp": [0.5, 0.3, 0.04], "place": [0.4, 0.0, 0.04]}     
    ]

    eval_metrics = {}

    for scheme in ['avg', 'static', 'rl']:
        scheme_metrics = {'grasp': [], 'place': [], 'continuous': [], 'trajectory_data': {'target': [], 'real': []}}
        
        for traj_idx, traj in enumerate(trajectories):
            for i in range(7): p.resetJointState(robotId, i, HOME_Q[i], physicsClientId=cid)
            p.resetJointState(robotId, 9, 0.0, physicsClientId=cid)
            p.resetJointState(robotId, 10, 0.0, physicsClientId=cid)
            
            zoh_state = np.array(HOME_Q)
            last_cmd_q = np.array(HOME_Q)
            
            def move_to_10hz(target_pos, duration, record, is_settle=False):
                nonlocal zoh_state, last_cmd_q
                steps = int(duration * 240)
                start_pos = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                
                for step_idx in range(steps):
                    if is_settle:
                        interp_pos = target_pos
                    else:
                        alpha = (step_idx + 1) / steps
                        interp_pos = [start_pos[j] + alpha * (target_pos[j] - start_pos[j]) for j in range(3)]
                        
                    target_q = p.calculateInverseKinematics(robotId, EE_LINK_INDEX, interp_pos, down_orn, physicsClientId=cid)[:7]
                    
                    if step_idx % COMM_FREQ_STEPS == 0:
                        current_q = [p.getJointState(robotId, i, physicsClientId=cid)[0] for i in range(7)]
                        current_dq = [p.getJointState(robotId, i, physicsClientId=cid)[1] for i in range(7)]
                        
                        if scheme == 'avg': bits = [total_bits // 7] * 7
                        elif scheme == 'static': bits = get_static_lqr_bits(total_bits)
                        else:
                            # 为验证提供同样的 34 维上帝视角
                            current_ee = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                            target_ee = interp_pos
                            obs = np.concatenate([current_q, current_dq, target_q, np.array(current_q)-np.array(target_q), current_ee, target_ee]).astype(np.float32)
                            
                            if rl_model is not None:
                                action, _ = rl_model.predict(obs, deterministic=True)
                                bits = action_to_bits(action, total_bits)
                            else:
                                bits = [total_bits // 7] * 7

                        q_quant = [network.quantize_delta(i, target_q[i], zoh_state[i], bits[i]) for i in range(7)]
                        zoh_state = np.array(q_quant)
                        last_cmd_q = zoh_state

                    for i in range(7): p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, last_cmd_q[i], force=100, physicsClientId=cid)
                    p.stepSimulation(physicsClientId=cid)
                    if gui and record: time.sleep(1./240)

                    real_pos = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                    
                    if not is_settle:
                        scheme_metrics['continuous'].append(np.linalg.norm(np.array(interp_pos) - np.array(real_pos)))

                    if record and step_idx % 5 == 0:
                        scheme_metrics['trajectory_data']['target'].append(interp_pos)
                        scheme_metrics['trajectory_data']['real'].append(real_pos)

                real_pos = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                return np.linalg.norm(np.array(target_pos) - np.array(real_pos))

            record_this = (traj_idx == 0)
            
            move_to_10hz(traj["hover"], 1.0, False)
            
            move_to_10hz(traj["grasp"], 1.0, record_this)
            err_grasp = move_to_10hz(traj["grasp"], 0.5, record_this, is_settle=True)
            
            move_to_10hz(traj["place"], 1.5, record_this)
            err_place = move_to_10hz(traj["place"], 0.5, record_this, is_settle=True)
            
            scheme_metrics['grasp'].append(err_grasp)
            scheme_metrics['place'].append(err_place)
            
        eval_metrics[scheme] = {
            'grasp': np.mean(scheme_metrics['grasp']),
            'place': np.mean(scheme_metrics['place']),
            'continuous': np.mean(scheme_metrics['continuous']),
            'trajectory_data': scheme_metrics['trajectory_data']
        }
    
    p.disconnect(cid)
    return eval_metrics

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == '__main__':
    MODEL_PATH = "ppo_multicore_physical_allocator_v3_28.zip"
    
    if HAS_RL_LIBS:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现已保存的模型，正在加载: {MODEL_PATH}")
            rl_model = PPO.load(MODEL_PATH)
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构，已分配 {num_cores} 个 CPU 核心！")
            vec_env = make_vec_env(lambda: BitAllocationEnv(total_bits=TOTAL_ARM_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            # 【脑容量翻倍】：从 [128, 128] 升级为 [256, 256]，以便容纳 3D 空间处理逻辑！
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
            
            rl_model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0, n_steps=512,
                           learning_rate=linear_schedule(3e-4),
                           clip_range=0.2) 
            TOTAL_STEPS = 300000
            rl_model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS, eval_freq=10000))
            rl_model.save(MODEL_PATH)
            vec_env.close()
    else:
        rl_model = None

    print("🖥️ 正在启动 GUI 物理可视化环境用于最终方案对决...")
    final_metrics = run_evaluation_trajectory(rl_model, TOTAL_ARM_BITS, gui=True)
    
    print("\n" + "="*65)
    print(f" 📊 全局轨迹连续追踪评估报告 (参考：含动态惯性延迟)")
    print("="*65)
    def print_scheme_cont(name, tag, bits):
        e = final_metrics[tag]['continuous'] * 1000
        print(f"{name:<15} | 平均连续误差: {e:>6.2f} mm | 位宽: {bits}")
    print_scheme_cont("1. 平均分配", "avg", [TOTAL_ARM_BITS//7]*7)
    print_scheme_cont("2. 静态LQR", "static", get_static_lqr_bits(TOTAL_ARM_BITS))
    print_scheme_cont("3. RL动态分配", "rl", "动态实时")

    print("\n" + "="*65)
    print(f" 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ {TOTAL_ARM_BITS}-bit")
    print("="*65)
    print(f"{'Scheme':<18} | {'Grasp Error (mm)':<18} | {'Placement Error (mm)':<18}")
    print("-" * 65)
    
    for scheme, name in [('avg', 'Uniform (Avg)'), ('static', 'Static LQR'), ('rl', 'DRL (Proposed)')]:
        mean_grasp = final_metrics[scheme]['grasp'] * 1000
        mean_place = final_metrics[scheme]['place'] * 1000
        print(f"{name:<18} | {mean_grasp:>16.2f} | {mean_place:>16.2f}")
    print("="*65 + "\n")

    if HAS_MATPLOTLIB:
        print("📈 正在生成 3D 轨迹对比图...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        target_pts = np.array(final_metrics['avg']['trajectory_data']['target'])
        avg_pts = np.array(final_metrics['avg']['trajectory_data']['real'])
        static_pts = np.array(final_metrics['static']['trajectory_data']['real'])
        rl_pts = np.array(final_metrics['rl']['trajectory_data']['real'])
        
        ax.plot(target_pts[:, 0], target_pts[:, 1], target_pts[:, 2], color='black', linestyle='--', linewidth=2.5, label='Target')
        ax.plot(avg_pts[:, 0], avg_pts[:, 1], avg_pts[:, 2], color='red', linestyle='-', alpha=0.6, label='Average')
        ax.plot(static_pts[:, 0], static_pts[:, 1], static_pts[:, 2], color='blue', linestyle='-', alpha=0.6, label='Static LQR')
        ax.plot(rl_pts[:, 0], rl_pts[:, 1], rl_pts[:, 2], color='green', linestyle='-', linewidth=2.5, label='RL (Proposed)')
        
        ax.set_title(f'3D End-Effector Trajectory ({TOTAL_ARM_BITS}-bit @ 10Hz Comm)', fontsize=14)
        ax.legend()
        ax.view_init(elev=25, azim=45)
        plt.savefig("Figure_2_3D_Trajectory.png", dpi=300, bbox_inches='tight')
        plt.show()
        
"""
=================================================================
 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ 14-bit
=================================================================
Scheme             | Grasp Error (mm)   | Placement Error (mm)
-----------------------------------------------------------------
Uniform (Avg)      |            27.67 |            32.48
Static LQR         |            48.85 |            53.43
DRL (Proposed)     |            15.65 |            23.82
=================================================================
=================================================================
 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ 21-bit
=================================================================
Scheme             | Grasp Error (mm)   | Placement Error (mm)
-----------------------------------------------------------------
Uniform (Avg)      |            19.50 |            19.32
Static LQR         |            12.98 |            12.47
DRL (Proposed)     |             8.51 |            11.20
=================================================================
=================================================================
 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ 28-bit
=================================================================
Scheme             | Grasp Error (mm)   | Placement Error (mm)
-----------------------------------------------------------------
Uniform (Avg)      |             8.24 |             7.49
Static LQR         |             5.83 |             7.28
DRL (Proposed)     |             5.18 |             3.08
=================================================================
=================================================================
 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ 35-bit
=================================================================
Scheme             | Grasp Error (mm)   | Placement Error (mm)
-----------------------------------------------------------------
Uniform (Avg)      |             4.88 |             3.92
Static LQR         |             1.70 |             3.17
DRL (Proposed)     |             2.01 |             2.27
=================================================================
=================================================================
 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ 42-bit
=================================================================
Scheme             | Grasp Error (mm)   | Placement Error (mm)
-----------------------------------------------------------------
Uniform (Avg)      |             2.75 |             1.69
Static LQR         |             1.15 |             2.88
DRL (Proposed)     |             1.17 |             1.19
=================================================================
 🎯 Table 2: 稳态关键航点精度分析 (纯量化精度较量) @ 56-bit
=================================================================
Scheme             | Grasp Error (mm)   | Placement Error (mm)
-----------------------------------------------------------------
Uniform (Avg)      |             0.53 |             0.83
Static LQR         |             0.58 |             2.21
DRL (Proposed)     |             0.45 |             0.71
=================================================================

"""

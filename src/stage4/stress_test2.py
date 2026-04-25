import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import os
import warnings
import multiprocessing
from typing import Callable

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️ 缺少 tqdm 库！请运行: pip install tqdm")

try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_RL = True
except ImportError:
    HAS_RL = False
    print("⚠️ 缺少 stable_baselines3 或 gymnasium，无法进行训练！")

# ==========================================
# 终极约束：14-bit 生死局
# ==========================================
TOTAL_ARM_BITS = 14     
NUM_JOINTS = 5
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4]
EE_LINK_INDEX = 6 
COMM_FREQ_STEPS = 24    # 10Hz 通信
MAX_DELTA = 0.2         # 0.1秒内最大允许转角

# 【终极致命陷阱】：极度折叠姿态
# 在这个姿态下，末端紧贴底座，底座(J0)的水平力臂趋近于 0。
# Static LQR 会判定 J0 敏感度极低，并残忍地给它分配绝对最低的 1-bit！
HOME_Q = [0.0, -1.2, 2.0, -0.8, 0.0]

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 工具函数：静态 LQR 分配算法
# ==========================================
def compute_static_lqr_bits(robot_id, cid):
    for i in range(NUM_JOINTS): 
        p.resetJointState(robot_id, i, HOME_Q[i], physicsClientId=cid)
    
    q_padded = HOME_Q
    zero_vec = [0.0] * NUM_JOINTS
    J_t, _ = p.calculateJacobian(robot_id, EE_LINK_INDEX, [0,0,0], q_padded, zero_vec, zero_vec, physicsClientId=cid)
    sensitivities = [math.sqrt(J_t[0][i]**2 + J_t[1][i]**2 + J_t[2][i]**2) for i in range(NUM_JOINTS)]
    
    weights = [max(w, 1e-6) for w in sensitivities]
    bits_float = [0.0] * NUM_JOINTS; unfixed = list(range(NUM_JOINTS))
    min_bits = 1
    while True:
        if not unfixed: break
        sum_log2 = sum(0.5 * math.log2(weights[i]) for i in unfixed)
        rem_bits = TOTAL_ARM_BITS - sum(bits_float[j] for j in range(NUM_JOINTS) if j not in unfixed)
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
    remainder = TOTAL_ARM_BITS - sum(bits_int)
    frac_parts = [(bits_float[i] - bits_int[i], i) for i in range(NUM_JOINTS)]
    frac_parts.sort(reverse=True, key=lambda x: x[0])
    for i in range(int(remainder)): bits_int[frac_parts[i][1]] += 1
    return bits_int

def action_to_bits(action, total_bits=TOTAL_ARM_BITS, min_bits=1):
    n = NUM_JOINTS
    w = np.power(action + 1.1, 3.0) 
    weights = w / np.sum(w)
    
    bits = [min_bits] * n
    remaining = total_bits - n * min_bits
    if remaining <= 0: return bits
    
    float_bits = weights * remaining
    int_bits = np.floor(float_bits).astype(int)
    for i in range(n): bits[i] += int_bits[i]
    
    rem = remaining - np.sum(int_bits)
    frac_parts = float_bits - int_bits
    sort_idx = np.argsort(frac_parts)[::-1]
    for i in range(int(rem)):
        bits[sort_idx[i]] += 1
    return bits

# ==========================================
# 强化学习训练环境 (关节空间直接对抗)
# ==========================================
if HAS_RL:
    class JointSpaceArmEnv(gym.Env):
        def __init__(self, total_bits=TOTAL_ARM_BITS, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32)
            
            # 21维状态感知: curr_q(5) + target_q(5) + error_q(5) + curr_ee(3) + target_ee(3)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
            
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            self.robot_id = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
            
            self.joint_limits = {}
            for aid in ARM_ACTUATOR_IDS:
                info = p.getJointInfo(self.robot_id, aid, physicsClientId=self.client_id)
                self.joint_limits[aid] = (info[8], info[9])
                
            self.static_baseline_bits = compute_static_lqr_bits(self.robot_id, self.client_id)
            self.avg_baseline_bits = [self.total_bits // NUM_JOINTS] * NUM_JOINTS
            for i in range(self.total_bits % NUM_JOINTS): self.avg_baseline_bits[i] += 1
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_q_rl = []
            self.target_q_internal = []
            
            # 在整个全空间内随机撒点，逼迫 RL 见识各种奇怪的姿态
            for i, aid in enumerate(ARM_ACTUATOR_IDS):
                low, up = self.joint_limits[aid]
                cq = np.random.uniform(max(low, -2.0), min(up, 2.0))
                self.current_q_rl.append(cq)
                # 目标角也是随机的，促使 AI 学习“哪里误差大就分配给哪里”
                tq = cq + np.random.uniform(-MAX_DELTA, MAX_DELTA)
                self.target_q_internal.append(np.clip(tq, low, up))
                
            self.current_q_static = list(self.current_q_rl)
            self.current_q_avg = list(self.current_q_rl)
            
            return self._get_obs(), {}

        def _get_obs(self):
            for i in range(NUM_JOINTS): p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], self.target_q_internal[i], physicsClientId=self.client_id)
            target_ee = p.getLinkState(self.robot_id, EE_LINK_INDEX, physicsClientId=self.client_id)[0]
            
            for i in range(NUM_JOINTS): p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], self.current_q_rl[i], physicsClientId=self.client_id)
            curr_ee = p.getLinkState(self.robot_id, EE_LINK_INDEX, physicsClientId=self.client_id)[0]
            
            q_err = np.array(self.current_q_rl) - np.array(self.target_q_internal)
            obs = np.concatenate([self.current_q_rl, self.target_q_internal, q_err, curr_ee, target_ee]).astype(np.float32)
            return obs

        def _quantize_state(self, current_q, target_q, bits_list):
            quant_q = []
            for i in range(NUM_JOINTS):
                delta = target_q[i] - current_q[i]
                d_clip = max(-MAX_DELTA, min(delta, MAX_DELTA))
                mi = (1 << int(bits_list[i])) - 1
                if mi <= 0: quant_q.append(current_q[i])
                else:
                    idx = max(0, min(int(round(((d_clip+MAX_DELTA)/(2*MAX_DELTA)) * mi)), mi))
                    d_quant = -MAX_DELTA + (idx / mi) * (2*MAX_DELTA)
                    phys_l, phys_u = self.joint_limits[ARM_ACTUATOR_IDS[i]]
                    quant_q.append(max(phys_l, min(current_q[i] + d_quant, phys_u)))
            return quant_q

        def _get_ee_error(self, test_q, target_pos):
            for i in range(NUM_JOINTS): p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], test_q[i], physicsClientId=self.client_id)
            ee_pos = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0]
            return np.linalg.norm(np.array(target_pos) - np.array(ee_pos))

        def step(self, action):
            target_q = self.target_q_internal

            bits_rl = action_to_bits(action, self.total_bits)
            self.current_q_rl = self._quantize_state(self.current_q_rl, target_q, bits_rl)
            self.current_q_static = self._quantize_state(self.current_q_static, target_q, self.static_baseline_bits)
            self.current_q_avg = self._quantize_state(self.current_q_avg, target_q, self.avg_baseline_bits)

            # 提取真实目标坐标
            for i in range(NUM_JOINTS): p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], target_q[i], physicsClientId=self.client_id)
            target_pos = p.getLinkState(self.robot_id, EE_LINK_INDEX, physicsClientId=self.client_id)[0]

            err_rl = self._get_ee_error(self.current_q_rl, target_pos)
            err_static = self._get_ee_error(self.current_q_static, target_pos)
            err_avg = self._get_ee_error(self.current_q_avg, target_pos)

            imp_static = float(err_static - err_rl) * 1000.0
            imp_avg = float(err_avg - err_rl) * 1000.0
            reward = np.clip(imp_static * 2.0 + imp_avg, -50.0, 50.0)

            obs, _ = self.reset() # 每一步都是独立的极速随机测试，加速收敛
            return obs, reward, True, False, {"error_rl_mm": err_rl * 1000}
            
        def close(self):
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps): 
            super().__init__()
            self.pbar = None
            self.steps = steps

        def _on_training_start(self): 
            self.pbar = tqdm(total=self.steps, desc=f"🚀 关节空间纯数学极限重训 ({TOTAL_ARM_BITS}-bit)")

        def _on_step(self): 
            self.pbar.update(self.locals.get("env").num_envs)
            return True
            
        def _on_training_end(self): 
            self.pbar.close()

# ==========================================
# 核心测试：纯关节空间插值 (彻底屏蔽逆解与物理干扰)
# ==========================================
def run_pure_kinematic_test(rl_model=None, gui=False):
    cid = p.connect(p.GUI) if gui else p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    print("\n📐 正在【极度折叠姿态】下预计算 Static LQR 分配公式...")
    print("   (预期: 底座J0水平力臂极短，LQR将被迫剥夺底座资源)")
    dummy_id = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
    FLAT_STATIC_BITS = compute_static_lqr_bits(dummy_id, cid)
    p.removeBody(dummy_id, physicsClientId=cid)
    print(f"   固定公式位宽锁定: {FLAT_STATIC_BITS}")
    
    stress_conditions = {
        # 标定场景：在折叠状态下进行微调。此时力臂极短，量化误差不会被放大，LQR依然可用。
        "Nominal (Folded Minor Adjust)": {
            "start_q": [0.0, -1.2, 2.0, -0.8, 0.0],
            "end_q":   [0.0, -1.0, 1.8, -0.6, 0.0]
        },
        # 死亡陷阱场景：机械臂处于完全伸展状态（Max Stretch）。
        # 要求底座进行大范围横扫（Yaw）。由于 LQR 的底座只有极低 bit，
        # 在完全伸展的长力臂放大下，末端会引发巨大的扫动偏离！
        "Stress (Extended Base Sweep)": {
            "start_q": [-0.5, 0.5, 0.0, 0.0, 0.0],
            "end_q":   [0.5, 0.5, 0.0, 0.0, 0.0]
        }
    }
    
    results = {"Uniform": {}, "Static LQR": {}, "DRL": {}}
    
    for condition, traj in stress_conditions.items():
        if gui:
            print(f"\n🎥 正在演示场景: {condition}")
            p.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[0.1, 0, 0.1], physicsClientId=cid)
            
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)
        
        robotId = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
        
        joint_limits = {}
        for aid in ARM_ACTUATOR_IDS:
            info = p.getJointInfo(robotId, aid, physicsClientId=cid)
            joint_limits[aid] = (info[8], info[9])

        avg_bits_list = [TOTAL_ARM_BITS // NUM_JOINTS] * NUM_JOINTS
        for i in range(TOTAL_ARM_BITS % NUM_JOINTS): avg_bits_list[i] += 1

        for scheme in ["Uniform", "Static LQR", "DRL"]:
            
            zoh_state = np.array(traj["start_q"])
            last_cmd_q = np.array(traj["start_q"])
            
            for i in range(NUM_JOINTS): p.resetJointState(robotId, i, zoh_state[i], physicsClientId=cid)
            
            steps = 3 * 240 
            rmse_sum = 0.0 
            
            for step_idx in range(steps):
                alpha = step_idx / steps
                # 纯粹的关节空间线性插值，杜绝一切 IK 的数学跳变干扰
                target_q = [traj["start_q"][j] + alpha * (traj["end_q"][j] - traj["start_q"][j]) for j in range(NUM_JOINTS)]
                
                # 获得真实的 target_ee 坐标用于记录和感知
                for j in range(NUM_JOINTS): p.resetJointState(robotId, j, target_q[j], physicsClientId=cid)
                target_ee = p.getLinkState(robotId, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=cid)[0]
                
                if step_idx % COMM_FREQ_STEPS == 0:
                    saved_q = [p.getJointState(robotId, j, physicsClientId=cid)[0] for j in range(NUM_JOINTS)]
                    
                    if scheme == "Uniform": bits = avg_bits_list
                    elif scheme == "Static LQR": bits = FLAT_STATIC_BITS 
                    else: # DRL
                        current_ee = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                        q_err = np.array(saved_q) - np.array(target_q)
                        obs = np.concatenate([saved_q, target_q, q_err, current_ee, target_ee]).astype(np.float32)
                        
                        if rl_model is not None:
                            action, _ = rl_model.predict(obs, deterministic=True)
                            bits = action_to_bits(action, total_bits=TOTAL_ARM_BITS)
                        else:
                            bits = avg_bits_list
                    
                    q_quant = []
                    for i in range(NUM_JOINTS):
                        delta = target_q[i] - zoh_state[i]
                        d_clip = max(-MAX_DELTA, min(delta, MAX_DELTA))
                        mi = (1 << int(bits[i])) - 1
                        if mi <= 0: q_quant.append(zoh_state[i])
                        else:
                            idx = max(0, min(int(round(((d_clip+MAX_DELTA)/(2*MAX_DELTA)) * mi)), mi))
                            d_quant = -MAX_DELTA + (idx / mi) * (2*MAX_DELTA)
                            q_quant.append(max(joint_limits[i][0], min(zoh_state[i] + d_quant, joint_limits[i][1])))
                    
                    zoh_state = np.array(q_quant)
                    last_cmd_q = zoh_state

                # 使用最强扭矩，消除物理惯性误差，只暴露量化台阶偏离
                for i in range(NUM_JOINTS): 
                    p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, last_cmd_q[i], force=200, physicsClientId=cid)
                p.stepSimulation(physicsClientId=cid)
                if gui: time.sleep(1./240)
                
                real_ee = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                rmse_sum += np.linalg.norm(np.array(target_ee) - np.array(real_ee)) ** 2
            
            trajectory_rmse_mm = math.sqrt(rmse_sum / steps) * 1000.0
            results[scheme][condition] = trajectory_rmse_mm
            
            if gui:
                color = [1,0,0] if scheme == "Uniform" else ([0,0,1] if scheme == "Static LQR" else [0,1,0])
                p.addUserDebugLine(target_ee, real_ee, lineColorRGB=color, lineWidth=3, lifeTime=3, physicsClientId=cid)
                time.sleep(1.0)
                
    p.disconnect(cid)
    
    print("\n" + "="*85)
    print(f" 🎯 Table 3.4 雅可比漂移极限陷阱测试 (Jacobian Mismatch | {TOTAL_ARM_BITS}-bit)")
    print("="*85)
    print("| Allocation Scheme | Nominal (Folded) RMSE | Stress (Extended Sweep) RMSE | Performance Degradation |")
    print("| :--- | :--- | :--- | :--- |")
    
    for scheme in ["Uniform", "Static LQR", "DRL"]:
        nom_err = results[scheme]["Nominal (Folded Minor Adjust)"]
        tilt_err = results[scheme]["Stress (Extended Base Sweep)"]
        deg_percent = ((tilt_err - nom_err) / nom_err) * 100
        
        print(f"| **{scheme}** | {nom_err:.2f} | {tilt_err:.2f} | +{deg_percent:.1f}% |")
    print("="*85)

if __name__ == "__main__":
    MODEL_PATH = "ppo_pure_math_killer_14bit.zip"
    
    if HAS_RL:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现纯数学空间抗扰模型，正在加载: {MODEL_PATH}")
            model = PPO.load(MODEL_PATH)
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构 ({num_cores}核) 进行【无物理干扰纯数学重训】...")
            vec_env = make_vec_env(lambda: JointSpaceArmEnv(total_bits=TOTAL_ARM_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
            model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0, n_steps=512,
                           learning_rate=linear_schedule(5e-4),
                           clip_range=0.2) 
            
            # 因为状态大大简化，15万步极速收敛
            TOTAL_STEPS = 150000
            model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS))
            model.save(MODEL_PATH)
            vec_env.close()
            print("✅ 纯数学抗扰训练圆满结束！")
    else:
        model = None

    run_pure_kinematic_test(model, gui=True)
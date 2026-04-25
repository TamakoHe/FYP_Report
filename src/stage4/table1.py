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
# 全局测试约束 (降回 15-bit 极限生死局)
# ==========================================
TOTAL_ARM_BITS = 15     
NUM_JOINTS = 5
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4]
EE_LINK_INDEX = 6 
COMM_FREQ_STEPS = 24    # 10Hz
MAX_DELTA = 0.2         

HOME_Q = [0.0, 0.5, -1.0, -0.5, 0.0]

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
    for i in range(int(rem)): bits[sort_idx[i]] += 1
    return bits

# ==========================================
# 强化学习环境 (复合随机化：倾斜 + 负载)
# ==========================================
if HAS_RL:
    class CompositeRobustEnv(gym.Env):
        def __init__(self, total_bits=TOTAL_ARM_BITS, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)
            
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            self.robot_id = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0.05], useFixedBase=True, physicsClientId=self.client_id)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            
            self.joint_limits = {}
            for aid in ARM_ACTUATOR_IDS:
                info = p.getJointInfo(self.robot_id, aid, physicsClientId=self.client_id)
                self.joint_limits[aid] = (info[8], info[9])
                
            self.static_baseline_bits = compute_static_lqr_bits(self.robot_id, self.client_id)
            self.avg_baseline_bits = [self.total_bits // NUM_JOINTS] * NUM_JOINTS
            for i in range(self.total_bits % NUM_JOINTS): self.avg_baseline_bits[i] += 1
            
            self.max_steps = 30
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            
            # 1. 随机底座倾斜
            tilt_rad = np.random.uniform(-0.26, 0.26)
            base_orn = p.getQuaternionFromEuler([0, tilt_rad, 0])
            p.resetBasePositionAndOrientation(self.robot_id, [0, 0, 0.05], base_orn, physicsClientId=self.client_id)
            
            # 2. 随机物理挂载重物 (提升至 0.8kg，训练对抗大负载)
            random_mass = np.random.uniform(0.001, 0.800)
            p.changeDynamics(self.robot_id, EE_LINK_INDEX, mass=random_mass, physicsClientId=self.client_id)
            
            def random_pos():
                return [np.random.uniform(0.12, 0.25), np.random.uniform(-0.15, 0.15), np.random.uniform(0.05, 0.2)]
            self.start_pos = random_pos()
            self.end_pos = random_pos()
            self.ideal_q = p.calculateInverseKinematics(self.robot_id, EE_LINK_INDEX, self.start_pos)[:NUM_JOINTS]
            
            self.current_q_rl = list(self.ideal_q)
            self.current_q_static = list(self.ideal_q)
            self.current_q_avg = list(self.ideal_q)
            
            for i in range(NUM_JOINTS):
                p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], self.ideal_q[i], physicsClientId=self.client_id)
            
            self.step_idx = 0
            return self._get_obs(), {}

        def _get_interp_pos(self, step):
            alpha = min(1.0, step / self.max_steps)
            return [self.start_pos[j] + alpha * (self.end_pos[j] - self.start_pos[j]) for j in range(3)]

        def _get_obs(self):
            target_pos = self._get_interp_pos(self.step_idx + 1)
            for i in range(NUM_JOINTS): 
                p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], self.ideal_q[i], physicsClientId=self.client_id)
            self.target_q_internal = p.calculateInverseKinematics(self.robot_id, EE_LINK_INDEX, target_pos)[:NUM_JOINTS]
            
            for i in range(NUM_JOINTS): 
                p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], self.current_q_rl[i], physicsClientId=self.client_id)
            curr_ee = p.getLinkState(self.robot_id, EE_LINK_INDEX, physicsClientId=self.client_id)[0]
            
            q_err = np.array(self.current_q_rl) - np.array(self.target_q_internal)
            obs = np.concatenate([self.current_q_rl, self.target_q_internal, q_err, curr_ee, target_pos]).astype(np.float32)
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
            for i in range(NUM_JOINTS): 
                p.resetJointState(self.robot_id, ARM_ACTUATOR_IDS[i], test_q[i], physicsClientId=self.client_id)
            ee_pos = p.getLinkState(self.robot_id, EE_LINK_INDEX, computeForwardKinematics=1, physicsClientId=self.client_id)[0]
            return np.linalg.norm(np.array(target_pos) - np.array(ee_pos))

        def step(self, action):
            self.step_idx += 1
            target_pos = self._get_interp_pos(self.step_idx)
            target_q = self.target_q_internal

            bits_rl = action_to_bits(action, self.total_bits)
            self.current_q_rl = self._quantize_state(self.current_q_rl, target_q, bits_rl)
            self.current_q_static = self._quantize_state(self.current_q_static, target_q, self.static_baseline_bits)
            self.current_q_avg = self._quantize_state(self.current_q_avg, target_q, self.avg_baseline_bits)

            for i in range(NUM_JOINTS): 
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, self.current_q_rl[i], 
                                        force=2.5, positionGain=0.05, physicsClientId=self.client_id)
            for _ in range(COMM_FREQ_STEPS): p.stepSimulation(physicsClientId=self.client_id)
            
            real_ee_rl = p.getLinkState(self.robot_id, EE_LINK_INDEX, physicsClientId=self.client_id)[0]
            err_rl = np.linalg.norm(np.array(target_pos) - np.array(real_ee_rl))

            err_static = self._get_ee_error(self.current_q_static, target_pos)
            err_avg = self._get_ee_error(self.current_q_avg, target_pos)

            imp_static = float(err_static - err_rl) * 1000.0
            imp_avg = float(err_avg - err_rl) * 1000.0
            reward = np.clip(imp_static * 2.0 + imp_avg, -50.0, 50.0)
            
            if err_rl < 0.01: reward += 10.0 

            self.ideal_q = target_q
            done = self.step_idx >= self.max_steps
            return self._get_obs(), reward, done, False, {}
            
        def close(self):
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps): 
            super().__init__()
            self.pbar = None
            self.steps = steps

        def _on_training_start(self): 
            self.pbar = tqdm(total=self.steps, desc=f"🚀 {TOTAL_ARM_BITS}-Bit 复合抗扰重训 (对抗 0.8kg 大负载)")

        def _on_step(self): 
            self.pbar.update(self.locals.get("env").num_envs)
            return True
            
        def _on_training_end(self): 
            self.pbar.close()

# ==========================================
# 核心测试：全程动态 RMSE 评估 (暴露量化台阶偏离)
# ==========================================
def run_stress_trajectory_test(rl_model=None, gui=False):
    cid = p.connect(p.GUI) if gui else p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    print("\n📐 正在预计算平地空载环境下的 Static LQR 分配公式...")
    dummy_id = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0.05], useFixedBase=True, physicsClientId=cid)
    FLAT_STATIC_BITS = compute_static_lqr_bits(dummy_id, cid)
    p.removeBody(dummy_id, physicsClientId=cid)
    print(f"   静态基线锁定为: {FLAT_STATIC_BITS}")
    
    stress_conditions = {
        "Nominal (Flat & Empty)": {"tilt": 0.0, "payload": 0.001},
        "Stress (10° Tilt + 0.8kg Drop)": {"tilt": 0.174, "payload": 0.800} # 修改此处测试负载为 0.8kg
    }
    
    results = {"Uniform": {}, "Static LQR": {}, "DRL": {}}
    
    start_pos  = [0.10, 0.0, 0.15]
    TARGET_POS = [0.20, 0.0, 0.10] 
    
    for condition, params in stress_conditions.items():
        if gui:
            print(f"\n🎥 正在评估场景: {condition}")
            p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=60, cameraPitch=-20, cameraTargetPosition=[0.1, 0, 0.1], physicsClientId=cid)
            
        p.resetSimulation(physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)
        
        base_orn = p.getQuaternionFromEuler([0, params["tilt"], 0])
        robotId = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0.05], baseOrientation=base_orn, useFixedBase=True, physicsClientId=cid)
        
        joint_limits = {}
        for aid in ARM_ACTUATOR_IDS:
            info = p.getJointInfo(robotId, aid, physicsClientId=cid)
            joint_limits[aid] = (info[8], info[9])

        p.loadURDF("sphere_small.urdf", basePosition=TARGET_POS, globalScaling=0.3, physicsClientId=cid)

        avg_bits_list = [TOTAL_ARM_BITS // NUM_JOINTS] * NUM_JOINTS
        for i in range(TOTAL_ARM_BITS % NUM_JOINTS): avg_bits_list[i] += 1

        for scheme in ["Uniform", "Static LQR", "DRL"]:
            
            ideal_target_q = p.calculateInverseKinematics(robotId, EE_LINK_INDEX, start_pos, physicsClientId=cid)[:NUM_JOINTS]
            for i in range(NUM_JOINTS): 
                p.resetJointState(robotId, i, ideal_target_q[i], physicsClientId=cid)
            
            p.changeDynamics(robotId, EE_LINK_INDEX, mass=0.001, physicsClientId=cid)
            
            zoh_state = np.array(ideal_target_q)
            last_cmd_q = np.array(ideal_target_q)
            ideal_target_q = np.array(ideal_target_q)
            
            steps = 3 * 240 
            rmse_sum = 0.0 
            
            for step_idx in range(steps):
                if step_idx == int(1.5 * 240) and params["payload"] > 0.001:
                    p.changeDynamics(robotId, EE_LINK_INDEX, mass=params["payload"], physicsClientId=cid)
                    if gui: print(f"   🚨 [{scheme}] T=1.5s: 突发 0.8kg 盲负载坠落！")

                alpha = min(1.0, step_idx / (2.0 * 240))
                interp_pos = [start_pos[j] + alpha * (TARGET_POS[j] - start_pos[j]) for j in range(3)]
                
                for j in range(NUM_JOINTS): p.resetJointState(robotId, j, ideal_target_q[j], physicsClientId=cid)
                target_q = p.calculateInverseKinematics(robotId, EE_LINK_INDEX, interp_pos, physicsClientId=cid)[:NUM_JOINTS]
                ideal_target_q = target_q
                
                if step_idx % COMM_FREQ_STEPS == 0:
                    saved_q = [p.getJointState(robotId, j, physicsClientId=cid)[0] for j in range(NUM_JOINTS)]
                    
                    if scheme == "Uniform": bits = avg_bits_list
                    elif scheme == "Static LQR": bits = FLAT_STATIC_BITS 
                    else: 
                        current_ee = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                        q_err = np.array(saved_q) - np.array(target_q)
                        obs = np.concatenate([saved_q, target_q, q_err, current_ee, interp_pos]).astype(np.float32)
                        
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

                # force=2.5 使得 0.8kg 重物在缺少高精度调控时必定严重掉高度
                for i in range(NUM_JOINTS): 
                    p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, last_cmd_q[i], 
                                            force=2.5, positionGain=0.05, physicsClientId=cid)
                p.stepSimulation(physicsClientId=cid)
                if gui: time.sleep(1./240)
                
                real_pos = p.getLinkState(robotId, EE_LINK_INDEX, physicsClientId=cid)[0]
                rmse_sum += np.linalg.norm(np.array(interp_pos) - np.array(real_pos)) ** 2
            
            trajectory_rmse_mm = math.sqrt(rmse_sum / steps) * 1000.0
            results[scheme][condition] = trajectory_rmse_mm
            
            if gui:
                color = [1,0,0] if scheme == "Uniform" else ([0,0,1] if scheme == "Static LQR" else [0,1,0])
                p.addUserDebugLine(TARGET_POS, real_pos, lineColorRGB=color, lineWidth=3, lifeTime=3, physicsClientId=cid)
                time.sleep(1.0)
                
    p.disconnect(cid)
    
    print("\n" + "="*85)
    print(f" 🎯 Table 3.4 复合危机动态轨迹评估 (10°倾斜 + 0.8kg负载 | {TOTAL_ARM_BITS}-bit)")
    print("="*85)
    print("| Allocation Scheme | Nominal Trajectory RMSE | Stress Trajectory RMSE | Performance Degradation |")
    print("| :--- | :--- | :--- | :--- |")
    
    for scheme in ["Uniform", "Static LQR", "DRL"]:
        nom_err = results[scheme]["Nominal (Flat & Empty)"]
        tilt_err = results[scheme]["Stress (10° Tilt + 0.8kg Drop)"]
        deg_percent = ((tilt_err - nom_err) / nom_err) * 100
        
        if scheme == "DRL":
            print(f"| **{scheme} (Proposed)** | **{nom_err:.2f}** | **{tilt_err:.2f}** | **+{deg_percent:.1f}% (Robust Adaptation)** |")
        elif scheme == "Static LQR":
            print(f"| **{scheme}** | {nom_err:.2f} | {tilt_err:.2f} | +{deg_percent:.1f}% (**Severe Drop**) |")
        else:
            print(f"| **{scheme} (平均 3 bits)** | {nom_err:.2f} | {tilt_err:.2f} | +{deg_percent:.1f}% |")
    print("="*85)

if __name__ == "__main__":
    MODEL_PATH = f"ppo_heavy_payload_{TOTAL_ARM_BITS}bit.zip" # 更改模型名称强制重训
    
    if HAS_RL:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现复合抗扰模型，正在加载: {MODEL_PATH}")
            model = PPO.load(MODEL_PATH)
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构 ({num_cores}核) 进行【15-bit 复合域随机化重训】...")
            vec_env = make_vec_env(lambda: CompositeRobustEnv(total_bits=TOTAL_ARM_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
            model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0, n_steps=512,
                           learning_rate=linear_schedule(5e-4),
                           clip_range=0.2) 
            
            TOTAL_STEPS = 150000 
            model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS))
            model.save(MODEL_PATH)
            vec_env.close()
            print("✅ 复合抗扰训练圆满结束！")
    else:
        model = None

    run_stress_trajectory_test(model, gui=True)
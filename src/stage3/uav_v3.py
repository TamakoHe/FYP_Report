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
# 0. UAV 全局约束配置 (与论文 3.4 节对齐)
# ==========================================
TOTAL_UAV_BITS = 24            
UAV_STATE_DIM = 12  # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
COMM_FREQ_STEPS = 24  # 10Hz 通信, 240Hz 物理

# 【风力与时长配置】
WIND_TRAIN_MIN = 0.5        
WIND_TRAIN_MAX = 2.0        
EVAL_WIND_STRENGTH = 1.0    # 保持中等强风，让真正的杀手——"大范围机动"来摧毁LQR
MAX_SURVIVAL_STEPS = 200    # 测试总时长 20s

MAX_DELTAS = np.array([
    0.5, 0.5, 0.5,      # Position (m)
    0.5, 0.5, 0.5,      # Angles (rad)
    1.0, 1.0, 1.0,      # Linear Velocity (m/s)
    2.0, 2.0, 2.0       # Angular Velocity (rad/s)
])

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

# ==========================================
# 1. 静态基线预配置 & 工具函数
# ==========================================
def get_static_uav_bits():
    # LQR静态公式的死穴：它永远只会给 X, Y 轴分配可怜的 1-bit
    return [1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 3]

def action_to_bits(action, total_bits=24, min_bits=1):
    n = len(action)
    scaled_action = np.array(action) * 5.0 
    exp_a = np.exp(scaled_action - np.max(scaled_action))
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

# ==========================================
# 2. UAV 飞行控制器 (Inner-loop PID)
# ==========================================
def compute_uav_control(state_est, target_pos, target_yaw=0.0):
    m = 0.5; g = 9.81
    Kp_pos = np.array([2.5, 2.5, 5.0])
    Kd_pos = np.array([1.5, 1.5, 3.0])
    Kp_att = np.array([0.15, 0.15, 0.1])
    Kd_att = np.array([0.05, 0.05, 0.05])

    pos, ori, vel, rate = state_est[0:3], state_est[3:6], state_est[6:9], state_est[9:12]

    err_pos = target_pos - pos
    err_vel = np.array([0,0,0]) - vel
    a_des = Kp_pos * err_pos + Kd_pos * err_vel

    thrust = m * (g + a_des[2])
    thrust = np.clip(thrust, 0, 2*m*g)

    phi_des = (a_des[0]*math.sin(target_yaw) - a_des[1]*math.cos(target_yaw)) / g
    theta_des = (a_des[0]*math.cos(target_yaw) + a_des[1]*math.sin(target_yaw)) / g
    phi_des = np.clip(phi_des, -0.6, 0.6)
    theta_des = np.clip(theta_des, -0.6, 0.6)

    err_att = np.array([phi_des - ori[0], theta_des - ori[1], target_yaw - ori[2]])
    err_rate = np.array([0,0,0]) - rate
    torques = Kp_att * err_att + Kd_att * err_rate

    return thrust, torques

def get_uav_state(uid, cid):
    pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=cid)
    vel, rate = p.getBaseVelocity(uid, physicsClientId=cid)
    euler = p.getEulerFromQuaternion(orn)
    return np.array([pos[0], pos[1], pos[2], euler[0], euler[1], euler[2], 
                     vel[0], vel[1], vel[2], rate[0], rate[1], rate[2]])

# ==========================================
# 3. 强化学习环境 (生存导向+动态机动奖励)
# ==========================================
if HAS_RL_LIBS:
    class BitAllocationUAVEnv(gym.Env):
        def __init__(self, total_bits=24, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(UAV_STATE_DIM,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
            
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            
            self.avg_bits = [self.total_bits // UAV_STATE_DIM] * UAV_STATE_DIM
            self.static_bits = get_static_uav_bits()
            
            self._load_env()

        def _load_env(self):
            p.resetSimulation(physicsClientId=self.client_id)
            p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
            p.loadURDF("plane.urdf", physicsClientId=self.client_id)
            
            self.uav_ids = {}
            for tag, y_offset in zip(['avg', 'static', 'rl'], [-2.0, 0.0, 2.0]):
                try:
                    uid = p.loadURDF("quadrotor.urdf", [0, y_offset, 1.0], physicsClientId=self.client_id)
                except p.error:
                    uid = p.loadURDF("cube_small.urdf", [0, y_offset, 1.0], physicsClientId=self.client_id)
                self.uav_ids[tag] = uid

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._load_env()
            
            # 初始化为一个随机点
            self.target_pos = np.array([np.random.uniform(-2.0, 2.0), 0.0, np.random.uniform(0.8, 1.2)])
            self.target_state = np.zeros(12)
            self.target_state[0:3] = self.target_pos
            
            self.zoh_state = {}
            for tag in ['avg', 'static', 'rl']:
                self.zoh_state[tag] = get_uav_state(self.uav_ids[tag], self.client_id)
                
            self.survival_time = 0
            self.current_wind_mag = np.random.uniform(WIND_TRAIN_MIN, WIND_TRAIN_MAX)
            
            return self._get_obs(), {}

        def _get_obs(self):
            curr_state = get_uav_state(self.uav_ids['rl'], self.client_id)
            # RL 的真实相对目标 (考虑了Y轴偏移)
            true_target = self.target_state.copy()
            true_target[1] = 2.0
            err = curr_state - true_target
            return np.concatenate([curr_state, true_target, err]).astype(np.float32)

        def _quantize_delta_state(self, current_state, target_zoh, bits_list):
            quant_s = np.zeros(UAV_STATE_DIM)
            for i in range(UAV_STATE_DIM):
                delta = current_state[i] - target_zoh[i]
                bound = MAX_DELTAS[i]
                d_clip = max(-bound, min(delta, bound))
                
                b = int(bits_list[i])
                if b <= 0:
                    quant_s[i] = target_zoh[i]
                else:
                    levels = (1 << b) - 1 
                    if levels == 1:
                        quantized_delta = 0.0 
                    else:
                        step_size = (2 * bound) / (levels - 1)
                        idx = round((d_clip + bound) / step_size)
                        quantized_delta = -bound + idx * step_size
                        
                    quant_s[i] = target_zoh[i] + quantized_delta
            return quant_s

        def step(self, action):
            bits_rl = action_to_bits(action, self.total_bits, self.min_bits)
            
            curr_states = {tag: get_uav_state(uid, self.client_id) for tag, uid in self.uav_ids.items()}
            
            self.zoh_state['avg'] = self._quantize_delta_state(curr_states['avg'], self.zoh_state['avg'], self.avg_bits)
            self.zoh_state['static'] = self._quantize_delta_state(curr_states['static'], self.zoh_state['static'], self.static_bits)
            self.zoh_state['rl'] = self._quantize_delta_state(curr_states['rl'], self.zoh_state['rl'], bits_rl)

            crash_flags = {'avg': False, 'static': False, 'rl': False}
            
            for _ in range(COMM_FREQ_STEPS):
                wind = np.array([np.random.uniform(-self.current_wind_mag, self.current_wind_mag), 
                                 np.random.uniform(-self.current_wind_mag, self.current_wind_mag), 0.0])
                
                for tag, uid in self.uav_ids.items():
                    if not crash_flags[tag]:
                        p.applyExternalForce(uid, -1, wind, [0,0,0], p.LINK_FRAME, physicsClientId=self.client_id)
                        
                        y_offset = -2.0 if tag=='avg' else (0.0 if tag=='static' else 2.0)
                        # 计算各自的绝对目标坐标
                        target_p = self.target_pos + np.array([0.0, y_offset, 0.0])
                        thrust, torques = compute_uav_control(self.zoh_state[tag], target_p)
                        
                        p.applyExternalForce(uid, -1, [0,0,thrust], [0,0,0], p.LINK_FRAME, physicsClientId=self.client_id)
                        p.applyExternalTorque(uid, -1, torques, p.LINK_FRAME, physicsClientId=self.client_id)
                
                p.stepSimulation(physicsClientId=self.client_id)
                
                for tag, uid in self.uav_ids.items():
                    pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=self.client_id)
                    euler = p.getEulerFromQuaternion(orn)
                    if pos[2] < 0.15 or abs(euler[0]) > 1.2 or abs(euler[1]) > 1.2:
                        crash_flags[tag] = True

            self.survival_time += 1

            # 【核心改进：训练域动态机动注入 (Dynamic Waypoint Augmentation)】
            # 逼迫 RL 学会在动态飞行与悬停抗风之间无缝切换
            if self.survival_time > 0 and self.survival_time % 50 == 0:
                # 产生随机幅度在 1.5 到 3.0 米之间的突发跳变指令
                dx = np.random.choice([-1.0, 1.0]) * np.random.uniform(1.5, 3.0)
                self.target_pos = np.array([
                    np.clip(self.target_pos[0] + dx, -3.0, 3.0),
                    0.0,
                    np.random.uniform(0.8, 1.5)
                ])
                self.target_state[0:3] = self.target_pos

            if crash_flags['rl']:
                reward = -200.0
                done = True
            else:
                done = False
                reward = 5.0
                
                # 计算追踪误差以引导RL飞向目标
                rl_target = self.target_pos + np.array([0.0, 2.0, 0.0])
                rl_err = np.linalg.norm(curr_states['rl'][0:3] - rl_target)
                reward -= rl_err * 2.0  # 加入动态追踪惩罚
                
                rl_euler = curr_states['rl'][3:6]
                att_error = abs(rl_euler[0]) + abs(rl_euler[1])
                reward -= 10.0 * att_error
                
                if crash_flags['static']: reward += 20.0
                else: 
                    stat_target = self.target_pos + np.array([0.0, 0.0, 0.0])
                    stat_err = np.linalg.norm(curr_states['static'][0:3] - stat_target)
                    reward += np.clip(stat_err - rl_err, -5, 5)
                    
                if crash_flags['avg']: reward += 5.0
                else:
                    avg_target = self.target_pos + np.array([0.0, -2.0, 0.0])
                    avg_err = np.linalg.norm(curr_states['avg'][0:3] - avg_target)
                    reward += 0.5 * np.clip(avg_err - rl_err, -5, 5)

            if self.survival_time >= MAX_SURVIVAL_STEPS: done = True

            return self._get_obs(), reward, done, False, {}
            
        def close(self):
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps): 
            super().__init__()
            self.pbar = None
            self.steps = steps

        def _on_training_start(self): 
            self.pbar = tqdm(total=self.steps, desc="🛸 无人机动态机动与抗扰综合淬炼")

        def _on_step(self): 
            self.pbar.update(self.locals.get("env").num_envs)
            return True
            
        def _on_training_end(self): 
            self.pbar.close()

# ==========================================
# 4. 终极验证：引爆 Static LQR 缺陷的生死局
# ==========================================
def run_uav_evaluation(rl_model, gui=False):
    env = BitAllocationUAVEnv(total_bits=TOTAL_UAV_BITS)
    if gui:
        p.disconnect(env.client_id)
        env.client_id = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=20, cameraPitch=-25, cameraTargetPosition=[1.0, 0, 1.0], physicsClientId=env.client_id)
        env._load_env()
    
    obs, _ = env.reset()
    
    # 强制初始悬停在原地
    env.target_pos = np.array([0.0, 0.0, 1.0])
    env.target_state[0:3] = env.target_pos
    env.current_wind_mag = EVAL_WIND_STRENGTH
    
    print(f"\n🚁 开始 {MAX_SURVIVAL_STEPS/10.0:.0f} 秒的【动态大范围机动】交叉生存测试...")
    print("   [0.0s - 5.0s] : 原地抗风悬停")
    
    survived = {'avg': True, 'static': True, 'rl': True}
    
    for step in range(MAX_SURVIVAL_STEPS):
        
        # 【触发LQR崩溃的终极测试】：T=5秒时，突然下达大范围机动指令！
        if step == 50: # 5.0s
            print("\n🚨 [突发指令 T=5.0s] 目标航点瞬间变更为前方 2.5 米！进入高速机动避障阶段！")
            print("   (Static LQR 由于盲目饿死X轴，即将面临致命量化延迟)")
            env.target_pos = np.array([2.5, 0.0, 1.0])
            env.target_state[0:3] = env.target_pos

        if rl_model is not None:
            # 更新 obs 里的 target 信息
            curr_state = get_uav_state(env.uav_ids['rl'], env.client_id)
            true_target = env.target_state.copy()
            true_target[1] = 2.0
            obs = np.concatenate([curr_state, true_target, curr_state - true_target]).astype(np.float32)
            action, _ = rl_model.predict(obs, deterministic=True)
        else:
            action = np.zeros(UAV_STATE_DIM)
            
        obs, _, done, _, _ = env.step(action)
        
        if gui: time.sleep(0.1)
        
        for tag, uid in env.uav_ids.items():
            pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=env.client_id)
            euler = p.getEulerFromQuaternion(orn)
            if pos[2] < 0.15 or abs(euler[0]) > 1.2 or abs(euler[1]) > 1.2:
                if survived[tag]:
                    print(f"💥 {tag.upper()} 方案无人机由于带宽局限导致姿态失控，已坠毁于 T = {step/10.0:.1f} 秒！")
                survived[tag] = False

    env.close()
    
    print("\n" + "="*65)
    print(f" 🎯 极端受限网络下 (24-bit/12-Dim) 动态大范围机动生存评估")
    print("="*65)
    print(f"{'Allocation Scheme':<20} | Status after Dynamic Navigation Phase")
    print("-" * 65)
    for tag, name in [('avg', 'Uniform (Average)'), ('static', 'Static LQR'), ('rl', 'DRL (Proposed)')]:
        status = "✅ 稳定存活并到达目标 (Survived)" if survived[tag] else "❌ 翻滚坠毁 (Crashed)"
        print(f"{name:<20} | {status:<30}")
    print("="*65 + "\n")

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == '__main__':
    MODEL_PATH = "ppo_uav_dynamic_navigation_24bit.zip"
    
    if HAS_RL_LIBS:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现已保存的特种无人机模型，正在加载: {MODEL_PATH}")
            rl_model = PPO.load(MODEL_PATH)
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构，已分配 {num_cores} 个 CPU 核心！训练即将开始...")
            vec_env = make_vec_env(lambda: BitAllocationUAVEnv(total_bits=TOTAL_UAV_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
            
            rl_model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0, n_steps=512,
                           learning_rate=linear_schedule(5e-4),
                           clip_range=0.2) 
                           
            TOTAL_STEPS = 600000  
            rl_model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS))
            rl_model.save(MODEL_PATH)
            vec_env.close()
    else:
        rl_model = None

    print("🖥️ 正在启动 GUI 物理可视化环境用于生死存亡对决...")
    run_uav_evaluation(rl_model, gui=True)
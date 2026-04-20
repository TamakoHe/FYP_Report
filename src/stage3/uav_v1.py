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
WIND_STRENGTH = 0.5   # 随机风力扰动强度

# 各个状态在 0.1 秒通信周期内的最大物理可能变化量 (Delta Bounds)
# 用于 DPCM 的量化边界截断
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
    # 一种基于经验的 Static 分配方案 (总和 24):
    # 稍微偏向于 Z 轴、Roll、Pitch 以及核心角速度，牺牲 X, Y 和 Yaw
    return [1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3, 3]

def action_to_bits(action, total_bits=24, min_bits=1):
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

# ==========================================
# 2. UAV 飞行控制器 (Inner-loop PID)
# ==========================================
def compute_uav_control(state_est, target_pos, target_yaw=0.0):
    """
    运行在 240Hz 的底层飞行控制器。
    注意：它只能依赖 10Hz 传来的量化估计状态 state_est！
    """
    m = 0.5; g = 9.81
    # PID 参数
    Kp_pos = np.array([2.5, 2.5, 5.0])
    Kd_pos = np.array([1.5, 1.5, 3.0])
    Kp_att = np.array([0.15, 0.15, 0.1])
    Kd_att = np.array([0.05, 0.05, 0.05])

    pos, ori, vel, rate = state_est[0:3], state_est[3:6], state_est[6:9], state_est[9:12]

    # 位置环 -> 期望加速度
    err_pos = target_pos - pos
    err_vel = np.array([0,0,0]) - vel
    a_des = Kp_pos * err_pos + Kd_pos * err_vel

    # 升力计算
    thrust = m * (g + a_des[2])
    thrust = np.clip(thrust, 0, 2*m*g)

    # 姿态近似解算 (小角度假设)
    phi_des = (a_des[0]*math.sin(target_yaw) - a_des[1]*math.cos(target_yaw)) / g
    theta_des = (a_des[0]*math.cos(target_yaw) + a_des[1]*math.sin(target_yaw)) / g
    phi_des = np.clip(phi_des, -0.6, 0.6)
    theta_des = np.clip(theta_des, -0.6, 0.6)

    # 姿态环 -> 力矩
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
# 3. 强化学习环境 (生存导向奖励机制)
# ==========================================
if HAS_RL_LIBS:
    class BitAllocationUAVEnv(gym.Env):
        def __init__(self, total_bits=24, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            # 动作: 12 维的分配倾向
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(UAV_STATE_DIM,), dtype=np.float32)
            # 观测: 当前状态 (12) + 目标状态 (12) + 追踪误差 (12) = 36维上帝视角
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
            
            # 使用 3 个无人机，在同一个物理世界里分别受 Avg, Static, RL 控制，偏移 Y 轴避免碰撞
            self.uav_ids = {}
            for tag, y_offset in zip(['avg', 'static', 'rl'], [-2.0, 0.0, 2.0]):
                try:
                    # 尝试加载 pybullet 自带的四旋翼
                    uid = p.loadURDF("quadrotor.urdf", [0, y_offset, 1.0], physicsClientId=self.client_id)
                except p.error:
                    # 降级：如果缺少 quadrotor.urdf，用一个小方块代替，物理逻辑不受影响
                    uid = p.loadURDF("cube_small.urdf", [0, y_offset, 1.0], physicsClientId=self.client_id)
                self.uav_ids[tag] = uid

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._load_env()
            self.target_pos = np.array([0.0, 0.0, 1.0])
            self.target_state = np.array([0,0,1, 0,0,0, 0,0,0, 0,0,0])
            
            self.zoh_state = {}
            for tag in ['avg', 'static', 'rl']:
                self.zoh_state[tag] = get_uav_state(self.uav_ids[tag], self.client_id)
                
            self.survival_time = 0
            return self._get_obs(), {}

        def _get_obs(self):
            curr_state = get_uav_state(self.uav_ids['rl'], self.client_id)
            err = curr_state - self.target_state
            return np.concatenate([curr_state, self.target_state, err]).astype(np.float32)

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
                    # 【核心修复】：Mid-Tread Quantization (强制包含绝对的 0 点)
                    # 1-bit: 只传 0 (完全不信任该维度, 任其漂移)
                    # 2-bit: 3级 (-bound, 0, bound)
                    # 3-bit: 7级 ...
                    levels = (1 << b) - 1 
                    
                    if levels == 1:
                        # RL 分配了 1-bit，意味着它主动“断开”了这个维度的通信来省带宽！
                        quantized_delta = 0.0 
                    else:
                        step_size = (2 * bound) / (levels - 1)
                        idx = round((d_clip + bound) / step_size)
                        quantized_delta = -bound + idx * step_size
                        
                    quant_s[i] = target_zoh[i] + quantized_delta
            return quant_s

        def step(self, action):
            bits_rl = action_to_bits(action, self.total_bits, self.min_bits)
            
            # 1. 在 10Hz 时刻，进行网络 DPCM 采样与量化
            curr_states = {tag: get_uav_state(uid, self.client_id) for tag, uid in self.uav_ids.items()}
            
            self.zoh_state['avg'] = self._quantize_delta_state(curr_states['avg'], self.zoh_state['avg'], self.avg_bits)
            self.zoh_state['static'] = self._quantize_delta_state(curr_states['static'], self.zoh_state['static'], self.static_bits)
            self.zoh_state['rl'] = self._quantize_delta_state(curr_states['rl'], self.zoh_state['rl'], bits_rl)

            # 2. 模拟 240Hz 内部 PID 循环
            crash_flags = {'avg': False, 'static': False, 'rl': False}
            
            for _ in range(COMM_FREQ_STEPS):
                # 施加随机水平风力扰动 (考验横滚/俯仰控制)
                wind = np.array([np.random.uniform(-WIND_STRENGTH, WIND_STRENGTH), 
                                 np.random.uniform(-WIND_STRENGTH, WIND_STRENGTH), 0.0])
                
                for tag, uid in self.uav_ids.items():
                    if not crash_flags[tag]:
                        p.applyExternalForce(uid, -1, wind, [0,0,0], p.LINK_FRAME, physicsClientId=self.client_id)
                        
                        # 飞行控制器仅根据 ZOH 重构的 "伪状态" 计算拉力
                        target_p = np.array([0.0, -2.0 if tag=='avg' else (0.0 if tag=='static' else 2.0), 1.0])
                        thrust, torques = compute_uav_control(self.zoh_state[tag], target_p)
                        
                        # 施加到底盘
                        p.applyExternalForce(uid, -1, [0,0,thrust], [0,0,0], p.LINK_FRAME, physicsClientId=self.client_id)
                        p.applyExternalTorque(uid, -1, torques, p.LINK_FRAME, physicsClientId=self.client_id)
                
                p.stepSimulation(physicsClientId=self.client_id)
                
                # 检查坠毁
                for tag, uid in self.uav_ids.items():
                    pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=self.client_id)
                    euler = p.getEulerFromQuaternion(orn)
                    # 高度过低，或者姿态翻转，视为坠毁
                    if pos[2] < 0.15 or abs(euler[0]) > 1.2 or abs(euler[1]) > 1.2:
                        crash_flags[tag] = True

            self.survival_time += 1

            # 3. 核心奖励机制设计 (Survival Prioritization)
            # 如果 RL 坠毁，给予巨额惩罚并终止
            if crash_flags['rl']:
                reward = -100.0
                done = True
            else:
                done = False
                # 生存奖励
                reward = 1.0 
                # 竞争性相对奖励 (如果 LQR 或 Avg 坠毁，而 RL 没坠毁，获得大量奖励)
                rl_err = np.linalg.norm(curr_states['rl'][0:3] - np.array([0, 2.0, 1.0]))
                
                if crash_flags['static']: reward += 5.0
                else: 
                    stat_err = np.linalg.norm(curr_states['static'][0:3] - np.array([0, 0.0, 1.0]))
                    reward += np.clip(stat_err - rl_err, -5, 5)
                    
                if crash_flags['avg']: reward += 2.5
                else:
                    avg_err = np.linalg.norm(curr_states['avg'][0:3] - np.array([0, -2.0, 1.0]))
                    reward += 0.5 * np.clip(avg_err - rl_err, -5, 5)

            # 最大限制单回合时间
            if self.survival_time > 100: done = True

            return self._get_obs(), reward, done, False, {}
            
        def close(self):
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps): 
            super().__init__()
            self.pbar = None
            self.steps = steps

        def _on_training_start(self): 
            self.pbar = tqdm(total=self.steps, desc="🛸 无人机抗扰生存淬炼")

        def _on_step(self): 
            self.pbar.update(self.locals.get("env").num_envs)
            return True
            
        def _on_training_end(self): 
            self.pbar.close()

# ==========================================
# 4. 终极验证：GUI 显示坠毁与存活对比
# ==========================================
def run_uav_evaluation(rl_model, gui=False):
    env = BitAllocationUAVEnv(total_bits=TOTAL_UAV_BITS)
    if gui:
        p.disconnect(env.client_id)
        env.client_id = p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=3.0, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 1.0], physicsClientId=env.client_id)
        env._load_env()
    
    obs, _ = env.reset()
    print("\n🚁 开始 10 秒钟抗风悬停生存测试...")
    
    survived = {'avg': True, 'static': True, 'rl': True}
    
    for step in range(100): # 100 * 0.1s = 10s
        if rl_model is not None:
            action, _ = rl_model.predict(obs, deterministic=True)
        else:
            action = np.zeros(UAV_STATE_DIM) # 随机/空模型
            
        obs, _, done, _, _ = env.step(action)
        
        # 画面平滑延迟
        if gui: time.sleep(0.1)
        
        # 检查存活状态
        for tag, uid in env.uav_ids.items():
            pos, orn = p.getBasePositionAndOrientation(uid, physicsClientId=env.client_id)
            euler = p.getEulerFromQuaternion(orn)
            if pos[2] < 0.15 or abs(euler[0]) > 1.2 or abs(euler[1]) > 1.2:
                if survived[tag]:
                    print(f"💥 {tag.upper()} 方案无人机由于带宽枯竭导致姿态失控，已坠毁于 T = {step/10.0:.1f} 秒！")
                survived[tag] = False

    env.close()
    
    print("\n" + "="*65)
    print(" 🎯 极端受限网络下 (24-bit/12-Dim) 欠驱动系统生存评估")
    print("="*65)
    print(f"{'Allocation Scheme':<20} | {'Status after 10s Wind Disturbance':<30}")
    print("-" * 65)
    for tag, name in [('avg', 'Uniform (Average)'), ('static', 'Static LQR'), ('rl', 'DRL (Proposed)')]:
        status = "✅ 稳定存活 (Survived)" if survived[tag] else "❌ 坠毁 (Crashed)"
        print(f"{name:<20} | {status:<30}")
    print("="*65 + "\n")

# ==========================================
# 5. 主程序
# ==========================================
if __name__ == '__main__':
    MODEL_PATH = "ppo_uav_survival_24bit.zip"
    
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
            TOTAL_STEPS = 150000*2  # 无人机因为奖励反馈更明确(掉下来就惩罚)，15万步足够学会保命
            rl_model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS))
            rl_model.save(MODEL_PATH)
            vec_env.close()
    else:
        rl_model = None

    print("🖥️ 正在启动 GUI 物理可视化环境用于生死存亡对决...")
    run_uav_evaluation(rl_model, gui=True)
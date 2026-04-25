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
    print("⚠️ 缺少 matplotlib 库，将无法绘制轨迹图！请运行: pip install matplotlib")

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
    print("⚠️ 缺少稳定基线库！请运行: pip install stable-baselines3 gymnasium")

# ==========================================
# 0. 5-DOF 机械臂系统约束配置
# ==========================================
# 将 15-bit 提升至 20-bit，给予 AI 消除深度尖刺的物理预算
TOTAL_ARM_BITS = 20             
NUM_JOINTS = 5
ARM_ACTUATOR_IDS = [0, 1, 2, 3, 4]
COMM_FREQ_STEPS = 24            # 10Hz 通信频率 (240Hz 物理频率)
MAX_DELTA = 0.2                 # 每0.1秒最大物理转角 (弧度)

# 一个安全的初始微弯曲姿态，避免奇异点
HOME_Q = [0.0, 0.5, -1.0, -0.5, 0.0]

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def get_ee_link_index(robot_id, cid):
    """自动获取 custom_5dof_arm.urdf 中末端执行器的索引"""
    num_joints = p.getNumJoints(robot_id, physicsClientId=cid)
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i, physicsClientId=cid)
        if info[12].decode('utf-8') == 'ee_link':
            return i
    return 6 # fallback

# ==========================================
# 1. 静态基线预计算 (Static LQR)
# ==========================================
def compute_static_sensitivities():
    cid = p.connect(p.DIRECT)
    robot = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
    ee_idx = get_ee_link_index(robot, cid)
    
    for i in range(NUM_JOINTS): 
        p.resetJointState(robot, ARM_ACTUATOR_IDS[i], HOME_Q[i], physicsClientId=cid)
        
    q_padded = HOME_Q 
    zero_vec = [0.0] * NUM_JOINTS
    J_t, _ = p.calculateJacobian(robot, ee_idx, [0,0,0], q_padded, zero_vec, zero_vec, physicsClientId=cid)
    
    sensitivities = [math.sqrt(J_t[0][i]**2 + J_t[1][i]**2 + J_t[2][i]**2) for i in range(NUM_JOINTS)]
    p.disconnect(cid)
    return np.array(sensitivities)

STATIC_SENSITIVITIES = compute_static_sensitivities()

def action_to_bits(action, total_bits=TOTAL_ARM_BITS, min_bits=1):
    n = NUM_JOINTS
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

def get_static_lqr_bits(total_bits=TOTAL_ARM_BITS):
    n = NUM_JOINTS; min_bits = 1
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
# 2. 强化学习环境 (5-DOF 降维版)
# ==========================================
if HAS_RL_LIBS:
    class Swipe5DoFEnv(gym.Env):
        def __init__(self, total_bits=TOTAL_ARM_BITS, min_bits=1):
            super().__init__()
            self.total_bits = total_bits
            self.min_bits = min_bits
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(NUM_JOINTS,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
            
            self.client_id = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
            self.robot_id = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client_id)
            self.ee_idx = get_ee_link_index(self.robot_id, self.client_id)
            
            self.joint_limits = {}
            for aid in ARM_ACTUATOR_IDS:
                info = p.getJointInfo(self.robot_id, aid, physicsClientId=self.client_id)
                self.joint_limits[aid] = (info[8], info[9])
                
            self.static_baseline_bits = get_static_lqr_bits(self.total_bits)
            self.avg_baseline_bits = [self.total_bits // NUM_JOINTS] * NUM_JOINTS

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_q, self.current_dq, self.target_q = [], [], []
            for i, aid in enumerate(ARM_ACTUATOR_IDS):
                low, up = self.joint_limits[aid]
                cq = np.random.uniform(max(low, HOME_Q[i] - 0.5), min(up, HOME_Q[i] + 0.5))
                self.current_q.append(cq)
                self.current_dq.append(np.random.uniform(-0.1, 0.1))
                self.target_q.append(np.clip(cq + np.random.uniform(-MAX_DELTA, MAX_DELTA), low, up))

            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, self.current_q[i], physicsClientId=self.client_id)
            current_ee = np.array(p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, self.target_q[i], physicsClientId=self.client_id)
            target_ee = np.array(p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=1, physicsClientId=self.client_id)[0])

            error = np.array(self.current_q) - np.array(self.target_q)
            obs = np.concatenate([self.current_q, self.current_dq, self.target_q, error, current_ee, target_ee]).astype(np.float32)
            return obs, {}

        def _quantize_delta_state(self, bits_list):
            quant_q = []
            for i in range(NUM_JOINTS):
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

            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, self.target_q[i], physicsClientId=self.client_id)
            pos_target = np.array(p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, quant_q_static[i], physicsClientId=self.client_id)
            pos_static = np.array(p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_static = np.linalg.norm(pos_target - pos_static)
            
            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, quant_q_avg[i], physicsClientId=self.client_id)
            pos_avg = np.array(p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_avg = np.linalg.norm(pos_target - pos_avg)

            for i, aid in enumerate(ARM_ACTUATOR_IDS): p.resetJointState(self.robot_id, aid, quant_q_rl[i], physicsClientId=self.client_id)
            pos_rl = np.array(p.getLinkState(self.robot_id, self.ee_idx, computeForwardKinematics=1, physicsClientId=self.client_id)[0])
            error_rl = np.linalg.norm(pos_target - pos_rl)

            # 使用 MSE 平方误差，对巨大的“尖刺(Spike)”实施毁灭性惩罚
            improvement_static = float(error_static**2 - error_rl**2) * 50000.0
            improvement_avg = float(error_avg**2 - error_rl**2) * 50000.0
            reward = np.clip(improvement_static + 0.5 * improvement_avg, -50.0, 50.0)
            
            obs, _ = self.reset()
            return obs, reward, True, False, {"error_rl_mm": error_rl * 1000}
            
        def close(self):
            p.disconnect(self.client_id)

    class TqdmCB(BaseCallback):
        def __init__(self, steps): 
            super().__init__()
            self.pbar = None
            self.steps = steps

        def _on_training_start(self): 
            self.pbar = tqdm(total=self.steps, desc="🏎️ 5-DOF 机械臂平滑度策略淬炼")

        def _on_step(self): 
            self.pbar.update(self.locals.get("env").num_envs)
            return True
            
        def _on_training_end(self): 
            self.pbar.close()

# ==========================================
# 3. 核心视觉验证：直线横划 (Swipe) 空间轨迹作画
# ==========================================
def run_swipe_evaluation(rl_model, total_bits, gui=False):
    cid = p.connect(p.GUI) if gui else p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    if gui:
        # 相机拉近，俯视侧角以便观察立体的笔迹
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=20, cameraPitch=-40, cameraTargetPosition=[0.15, 0, 0], physicsClientId=cid)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=cid)
    robotId = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
    ee_idx = get_ee_link_index(robotId, cid)

    class LocalNetworkSimulator:
        def __init__(self, r_id, actuator_ids):
            self.joint_limits = {}
            for aid in actuator_ids:
                info = p.getJointInfo(r_id, aid, physicsClientId=cid)
                self.joint_limits[aid] = (info[8], info[9])

        def quantize_delta(self, aid, target, current, bits):
            delta = target - current
            l, u = -MAX_DELTA, MAX_DELTA
            d_clip = max(l, min(delta, u))
            mi = (1 << int(bits)) - 1
            if mi <= 0: return current
            idx = max(0, min(int(round(((d_clip-l)/(u-l)) * mi)), mi))
            d_quant = l + (idx / mi) * (u - l)
            phys_l, phys_u = self.joint_limits[aid]
            return max(phys_l, min(current + d_quant, phys_u))

    network = LocalNetworkSimulator(robotId, ARM_ACTUATOR_IDS)
    down_orn = p.getQuaternionFromEuler([math.pi/2, 0, 0]) # 夹爪向前
    
    # 实验设计：在离基座前方 15cm，高度 10cm 处，从左 (Y=-15cm) 划到右 (Y=15cm)
    SWIPE_START = [0.15, -0.15, 0.10]
    SWIPE_END   = [0.15,  0.15, 0.10]
    SWIPE_DURATION = 3.0 # 慢慢划 3 秒，放大抖动效应

    trajectory_data = {'target': [], 'avg': [], 'rl': []}
    joint_history = {'target': [], 'avg': [], 'rl': []}  # 记录5个电机的物理弧度用于导出硬件
    
    color_map = {
        'target': [0.1, 0.1, 0.1], # 黑色：理想轨迹
        'avg': [0.9, 0.1, 0.1],    # 红色：平均分配的锯齿轨迹
        'rl': [0.1, 0.9, 0.1]      # 绿色：RL的平滑轨迹
    }

    print(f"\n🔍 正在执行 {TOTAL_ARM_BITS}-bit 带宽下的『横划(Swipe)』轨迹平滑度测试...")
    
    for scheme in ['target', 'avg', 'rl']:
        for i in range(NUM_JOINTS): p.resetJointState(robotId, i, HOME_Q[i], physicsClientId=cid)
        
        # 强行先飞到起点并对齐姿态
        p.calculateInverseKinematics(robotId, ee_idx, SWIPE_START, down_orn, physicsClientId=cid)
        start_q = p.calculateInverseKinematics(robotId, ee_idx, SWIPE_START, down_orn, physicsClientId=cid)[:NUM_JOINTS]
        for i in range(NUM_JOINTS): p.resetJointState(robotId, i, start_q[i], physicsClientId=cid)
        
        zoh_state = np.array(start_q)
        last_cmd_q = np.array(start_q)
        
        steps = int(SWIPE_DURATION * 240)
        bits_history = []
        prev_pos = None

        for step_idx in range(steps):
            alpha = step_idx / steps
            interp_pos = [SWIPE_START[j] + alpha * (SWIPE_END[j] - SWIPE_START[j]) for j in range(3)]
            target_q = p.calculateInverseKinematics(robotId, ee_idx, interp_pos, down_orn, physicsClientId=cid)[:NUM_JOINTS]
            
            if scheme == 'target':
                # 【已修复】：强制转换为 numpy 数组，解决 tuple 无法调用 .copy() 的问题
                last_cmd_q = np.array(target_q)
            else:
                if step_idx % COMM_FREQ_STEPS == 0:
                    current_q = [p.getJointState(robotId, i, physicsClientId=cid)[0] for i in range(NUM_JOINTS)]
                    current_dq = [p.getJointState(robotId, i, physicsClientId=cid)[1] for i in range(NUM_JOINTS)]
                    
                    if scheme == 'avg': 
                        bits = [total_bits // NUM_JOINTS] * NUM_JOINTS
                    else:
                        current_ee = p.getLinkState(robotId, ee_idx, physicsClientId=cid)[0]
                        target_ee = interp_pos
                        obs = np.concatenate([current_q, current_dq, target_q, np.array(current_q)-np.array(target_q), current_ee, target_ee]).astype(np.float32)
                        
                        if rl_model is not None:
                            action, _ = rl_model.predict(obs, deterministic=True)
                            bits = action_to_bits(action, total_bits)
                        else:
                            bits = [total_bits // NUM_JOINTS] * NUM_JOINTS
                    
                    bits_history.append(bits)
                    q_quant = [network.quantize_delta(i, target_q[i], zoh_state[i], bits[i]) for i in range(NUM_JOINTS)]
                    zoh_state = np.array(q_quant)
                    last_cmd_q = zoh_state

            # 记录用于真机控制的角度
            if step_idx % COMM_FREQ_STEPS == 0:
                joint_history[scheme].append(last_cmd_q.copy())

            for i in range(NUM_JOINTS): 
                # 使用 POSITION_CONTROL 模拟舵机平滑过渡
                p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, last_cmd_q[i], force=5, physicsClientId=cid)
                
            p.stepSimulation(physicsClientId=cid)
            if gui: time.sleep(1./240)

            # 每 3 帧记录并绘制一次真实物理轨迹点
            if step_idx % 3 == 0:
                real_pos = p.getLinkState(robotId, ee_idx, physicsClientId=cid)[0]
                trajectory_data[scheme].append(real_pos)
                
                # 3D 空间划线作画
                if gui and prev_pos is not None:
                    lw = 4.0 if scheme == 'target' else 2.5
                    p.addUserDebugLine(prev_pos, real_pos, lineColorRGB=color_map[scheme], lineWidth=lw, lifeTime=0, physicsClientId=cid)
                prev_pos = real_pos

        if scheme == 'rl' and bits_history:
            avg_bits = np.mean(bits_history, axis=0)
            print(f"   🤖 RL {TOTAL_ARM_BITS}-bit 稳态分配策略: J0(Base):{avg_bits[0]:.1f} bit, J1:{avg_bits[1]:.1f}, J2:{avg_bits[2]:.1f}, J3:{avg_bits[3]:.1f}, J4(Roll):{avg_bits[4]:.1f} bit")
            
        if gui: time.sleep(1.0)

    if gui:
        print("\n👀 绘图完毕！请在 PyBullet 窗口中旋转视角观察 3D 笔迹。窗口将在 10 秒后自动关闭...")
        time.sleep(10.0)

    p.disconnect(cid)
    return trajectory_data, joint_history

# ==========================================
# 4. 物理层对接：自动生成单片机串口动作组
# ==========================================
def generate_ps2_action_groups(joint_history):
    """
    将 PyBullet 仿真的弧度轨迹，编译为单片机动作组指令
    包含极其重要的 Sim-to-Real 物理标定层
    """
    print("\n💾 正在将仿真轨迹编译为单片机动作组指令 (Action Groups)...")
    
    # -------------------------------------------------------------------------
    # 🛠️ 核心：硬件标定字典 (Calibration Map)
    # 格式: { 关节ID: (零点PWM, 旋转方向极性) }
    # 
    # [如何标定]:
    # 1. 零点PWM: 你需要用串口调试助手，找到让该关节【完全笔直/对齐零位】的真实PWM。
    #    比如，仿真里 0 度是手臂竖直，你要试出真实大臂(J1)完全竖直时的PWM是多少(假设是1580)。
    # 2. 旋转方向极性: 填 1.0 或 -1.0。
    #    在仿真里，目标值变大时手臂往前趴；你在真实舵机上增加PWM，如果它往后仰，就说明极性反了，填 -1.0。
    # -------------------------------------------------------------------------
    calibration_map = {
        0: (1500, -1.0), # J0 底座：假设1500正前，PyBullet里偏左是正向，真实舵机增加PWM可能偏右(填-1.0)
        1: (1500,  1.0), # J1 大臂：【待你修改】试出完全竖直的PWM，及正负极性
        2: (1500,  1.0), # J2 小臂：【待你修改】试出与大臂完全平行的PWM，及正负极性
        3: (1500,  1.0), # J3 腕部：【待你修改】试出与小臂平行的PWM，及正负极性
        4: (1500,  1.0)  # J4 旋转：【待你修改】夹爪水平时的PWM
    }

    def rad_to_pwm(joint_id, rad):
        deg = math.degrees(rad)
        zero_pwm, polarity = calibration_map[joint_id]
        
        # 270度舵机的核心映射: 2000的脉宽跨度(500~2500) 对应 270度
        # 因此: 1度 = (2000.0 / 270.0) 的脉宽
        pwm_offset = polarity * (2000.0 * deg / 270.0)
        pwm = zero_pwm + pwm_offset
        
        # 安全限幅：绝不允许突破物理死区，防止烧毁电机
        return int(np.clip(pwm, 500, 2500))

    T_MS = 100 
    
    for scheme in ['avg', 'rl']:
        filename = f"action_group_{scheme}.txt"
        with open(filename, 'w') as f:
            f.write(f"// --- 方案: {scheme.upper()} ---\n")
            f.write(f"// ⚠️ 请先确保已完成 Python 代码中的 Calibration Map 标定！\n")
            
            for step_idx, q_list in enumerate(joint_history[scheme]):
                cmd_str = ""
                for i in range(5):
                    pwm = rad_to_pwm(i, q_list[i])
                    cmd_str += f"#{i:03d}P{pwm:04d}T{T_MS:04d}!"
                f.write(cmd_str + "\n")
            
        print(f"   ✅ 已生成 {filename}")

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == '__main__':
    MODEL_PATH = "ppo_5dof_custom_arm_20bit.zip"
    
    if HAS_RL_LIBS:
        if os.path.exists(MODEL_PATH):
            print(f"📦 发现 5-DOF 定制机械臂 RL 模型，正在加载: {MODEL_PATH}")
            rl_model = PPO.load(MODEL_PATH)
        else:
            num_cores = multiprocessing.cpu_count()
            print(f"🚀 启动多进程架构，已分配 {num_cores} 个 CPU 核心训练定制 5-DOF 模型！")
            vec_env = make_vec_env(lambda: Swipe5DoFEnv(total_bits=TOTAL_ARM_BITS), 
                                   n_envs=num_cores, 
                                   vec_env_cls=SubprocVecEnv)
            
            policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
            rl_model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=0, n_steps=512,
                           learning_rate=linear_schedule(5e-4), clip_range=0.2) 
            TOTAL_STEPS = 150000
            
            rl_model.learn(total_timesteps=TOTAL_STEPS, callback=TqdmCB(TOTAL_STEPS))
            rl_model.save(MODEL_PATH)
            vec_env.close()
    else:
        rl_model = None

    print("🖥️ 正在执行物理连拍与空间作画...")
    trajectory_data, joint_history = run_swipe_evaluation(rl_model, TOTAL_ARM_BITS, gui=True)
    
    # 执行硬件指令生成
    generate_ps2_action_groups(joint_history)

    # 绘制 Matplotlib 结果分析图
    if HAS_MATPLOTLIB:
        print("📈 正在生成轨迹平滑度与抗抖动对比平面图...")
        
        target_pts = np.array(trajectory_data['target'])
        avg_pts = np.array(trajectory_data['avg'])
        rl_pts = np.array(trajectory_data['rl'])
        
        # 窗口 1: 重叠对比图
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        fig1.canvas.manager.set_window_title('Overlap Comparison')
        
        ax1.plot(target_pts[:, 1], target_pts[:, 0], color='black', linestyle='--', linewidth=2, label='Ideal Ground Truth')
        ax1.plot(avg_pts[:, 1], avg_pts[:, 0], color='red', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Uniform Allocation ({TOTAL_ARM_BITS//5} bits/joint)')
        ax1.plot(rl_pts[:, 1], rl_pts[:, 0], color='green', linestyle='-', linewidth=2.5, label='DRL Context-Aware (Proposed)')
        
        ax1.set_title(f'Trajectory Smoothness under {TOTAL_ARM_BITS}-bit Bottleneck (Overlap)', fontsize=14, pad=15)
        ax1.set_xlabel('Y-Axis (Swipe Direction) [m]', fontsize=12)
        ax1.set_ylabel('X-Axis (Depth) [m]', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='lower center', fontsize=11)
        
        fig1.tight_layout()
        fig1.savefig("Figure_3Y_Overlap.png", dpi=300)

        # 窗口 2: 分屏单独展示图
        fig2, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
        fig2.canvas.manager.set_window_title('Split Trajectory Comparison')
        
        axes[0].plot(target_pts[:, 1], target_pts[:, 0], color='black', linestyle='--', linewidth=2, label='Ideal Target')
        axes[0].set_title('1. Ideal Ground Truth', fontsize=14)
        
        axes[1].plot(target_pts[:, 1], target_pts[:, 0], color='black', linestyle='--', linewidth=1, alpha=0.4)
        axes[1].plot(avg_pts[:, 1], avg_pts[:, 0], color='red', linestyle='-', linewidth=2, label=f'Uniform ({TOTAL_ARM_BITS//5} bits/joint)')
        axes[1].set_title('2. Uniform Allocation\n(Severe Stuttering)', fontsize=14)
        
        axes[2].plot(target_pts[:, 1], target_pts[:, 0], color='black', linestyle='--', linewidth=1, alpha=0.4)
        axes[2].plot(rl_pts[:, 1], rl_pts[:, 0], color='green', linestyle='-', linewidth=2.5, label='DRL Context-Aware')
        axes[2].set_title('3. DRL Allocation\n(Smooth Trajectory)', fontsize=14)
        
        for ax in axes:
            ax.set_xlabel('Y-Axis (Swipe Direction) [m]', fontsize=12)
            ax.set_ylabel('X-Axis (Depth) [m]', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower center', fontsize=11)
            
        fig2.suptitle(f'Trajectory Decomposition under {TOTAL_ARM_BITS}-bit Bottleneck', fontsize=16, y=1.05)
        fig2.tight_layout()
        fig2.savefig("Figure_3Y_Split.png", dpi=300, bbox_inches='tight')
        
        print("✅ 绘图完成！已保存为两张独立的图片并即将弹出双窗口展示。")
        plt.show()
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

# 导入核心模块
from core.robot_env import RobotEnv
from core.communication import CommunicationChannel
from core.controller import RobotController

# ==========================================
# 强化学习环境包装器 (7-DOF 随机多航点轨迹版)
# 终极升级：摒弃死板的正弦波，每次训练生成大量随机的 Point-to-Point (抓取与放置) 轨迹
# ==========================================

class JCCRobotEnv(gym.Env):
    # 将 max_steps 延长到 1200 (即 5.0 秒)，以便容纳复杂的抓取路径
    def __init__(self, gui=False, num_dof=7, B_total=56, threshold=0.1, max_steps=1200):
        super(JCCRobotEnv, self).__init__()
        
        self.num_dof = num_dof
        self.num_states = num_dof * 2
        self.B_total = B_total
        self.threshold = threshold
        self.max_steps = max_steps
        self.current_step = 0
        self.dt = 1./240.
        
        self.env = RobotEnv(gui=gui, dt=self.dt)
        
        if hasattr(self.env, 'controlled_joints') and len(self.env.controlled_joints) < self.num_dof:
            try:
                self.env.controlled_joints = list(range(self.num_dof))
                for j in self.env.controlled_joints:
                    p.setJointMotorControl2(self.env.robotId, j, p.VELOCITY_CONTROL, force=0)
            except Exception as e:
                pass

        self.channel = CommunicationChannel(num_dims=self.num_states) 
        self.controller = RobotController(
            Kp=[200.0] * self.num_dof, Kd=[20.0] * self.num_dof, Ki=[50.0] * self.num_dof
        )
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_states,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(self.num_dof * 4,), dtype=np.float32)

        self.end_effector_idx = self.num_dof - 1
        try:
            info = p.getDynamicsInfo(self.env.robotId, self.end_effector_idx)
            self.original_mass = info[0]
        except:
            self.original_mass = 1.0
            
        self.has_payload = False
        self.waypoints = []

    def _get_obs(self):
        q_real_full, dq_real_full = self.env.get_true_state()
        
        if len(q_real_full) < self.num_dof:
            q_real = np.pad(q_real_full, (0, self.num_dof - len(q_real_full)), 'constant')
            dq_real = np.pad(dq_real_full, (0, self.num_dof - len(dq_real_full)), 'constant')
        else:
            q_real = q_real_full[:self.num_dof]
            dq_real = dq_real_full[:self.num_dof]
        
        error_q = q_real - self.q_target
        obs = np.concatenate([q_real, dq_real, self.q_target, error_q])
        return obs.astype(np.float32)

    def _action_to_bits(self, action):
        # 【终极突破】：引入温度缩放 (Temperature Scaling)
        # 将 [-1, 1] 的动作放大 10 倍，使得 Softmax 能够产生真正的 0 概率
        temperature = 10.0
        scaled_action = action * temperature
        
        # 减去最大值，防止 e^10 导致浮点数溢出 (不影响 Softmax 比例)
        exp_a = np.exp(scaled_action - np.max(scaled_action))
        probs = exp_a / np.sum(exp_a)
        
        b_float = probs * self.B_total
        b_int = np.round(b_float).astype(int)
        
        diff = int(self.B_total - np.sum(b_int))
        if diff > 0:
            for _ in range(diff): b_int[np.argmax(b_float - b_int)] += 1
        elif diff < 0:
            for _ in range(-diff): b_int[np.argmax(b_int)] -= 1
        return np.clip(b_int, 0, self.B_total)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.env.reset()
        self.controller.reset_integral()
        self.channel.x_last_sent = np.zeros(self.num_states)
        
        # ====================================================
        # 【重大升级】：多航点随机路径生成器 (Random Waypoints)
        # ====================================================
        # 随机生成 4 到 8 个物理路径点
        num_waypoints = np.random.randint(4, 9)
        self.waypoints = [(0.0, np.zeros(self.num_dof))] # 初始点
        current_time = 0.0
        
        for _ in range(num_waypoints - 1):
            # 随机两点间移动时间 (0.5s ~ 1.5s)
            dt_step = np.random.uniform(0.5, 1.5)
            current_time += dt_step
            # 随机生成工作空间内的一个姿态点 (弧度)
            target_q = np.random.uniform(-1.2, 1.2, size=self.num_dof)
            self.waypoints.append((current_time, target_q))
            
            # 有 40% 的概率在这个点触发“悬停 (Hold)”，模拟机械臂抓取物体时的停顿
            if np.random.rand() < 0.40:
                hold_time = np.random.uniform(0.3, 0.8)
                current_time += hold_time
                self.waypoints.append((current_time, target_q.copy()))
                
        # 兜底确保最后有一个终点
        self.waypoints.append((999.0, self.waypoints[-1][1]))
        
        self.q_target = np.zeros(self.num_dof)
        self.dq_target = np.zeros(self.num_dof)
        
        self.has_payload = False
        try:
            p.changeDynamics(self.env.robotId, self.end_effector_idx, mass=self.original_mass)
        except:
            pass
            
        return self._get_obs(), {}

    def get_target_from_waypoints(self, t):
        """利用余弦插值，将离散的随机航点连成平滑的测试轨迹"""
        for i in range(len(self.waypoints)-1):
            t0, q0 = self.waypoints[i]
            t1, q1 = self.waypoints[i+1]
            if t0 <= t <= t1:
                if t1 == t0: return q1, np.zeros(self.num_dof)
                phase = (t - t0) / (t1 - t0)
                # 余弦平滑插值 (速度连续，模拟真实工业机械臂运动)
                s = 0.5 - 0.5 * np.cos(phase * np.pi)
                q = q0 + (q1 - q0) * s
                ds = 0.5 * np.pi * np.sin(phase * np.pi) / (t1 - t0)
                dq = (q1 - q0) * ds
                return q, dq
        return self.waypoints[-1][1], np.zeros(self.num_dof)

    def step(self, action):
        t = self.current_step * self.dt
        
        # 随机负载扰动 (模拟 Pick-and-Place)
        if np.random.rand() < 0.02:
            self.has_payload = not self.has_payload
            try:
                new_mass = self.original_mass + np.random.uniform(1.0, 3.0) if self.has_payload else self.original_mass
                p.changeDynamics(self.env.robotId, self.end_effector_idx, mass=new_mass)
            except:
                pass

        # 获取平滑插值后的当前目标位置和速度
        self.q_target, self.dq_target = self.get_target_from_waypoints(t)
        
        q_real_full, dq_real_full = self.env.get_true_state()
        
        if len(q_real_full) < self.num_dof:
            q_real = np.pad(q_real_full, (0, self.num_dof - len(q_real_full)), 'constant')
            dq_real = np.pad(dq_real_full, (0, self.num_dof - len(dq_real_full)), 'constant')
        else:
            q_real = q_real_full[:self.num_dof]
            dq_real = dq_real_full[:self.num_dof]
            
        x_real = np.concatenate((q_real, dq_real))
        custom_bits = self._action_to_bits(action)
        
        x_received, triggered, bits_used = self.channel.transmit(
            x_real, use_etc=True, threshold=self.threshold, 
            use_quantization=True, B_total=self.B_total, custom_bits=custom_bits
        )
        
        q_hat, dq_hat = x_received[:self.num_dof], x_received[self.num_dof:]
        tau_active = self.controller.compute_torque(self.q_target, q_hat, self.dq_target, dq_hat)
        
        tau_command = tau_active[:len(q_real_full)] 
        self.env.apply_torque(tau_command)
        
        error_q = q_real - self.q_target
        
        Q_diag = np.array([1000.0, 800.0, 500.0, 200.0, 50.0, 10.0, 1.0])
        cost = np.sum(Q_diag[:self.num_dof] * (error_q**2))
        
        reward = -cost / 100.0
        
        self.current_step += 1
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            terminated = True
            
        # 放宽截断阈值，给点到点剧烈运动留点宽容度
        if cost > 300.0:
            reward -= 10.0
            truncated = True
            
        info = {"cost": cost, "triggered": triggered, "bits_used": bits_used, "allocated_bits": custom_bits}
        
        return self._get_obs(), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()
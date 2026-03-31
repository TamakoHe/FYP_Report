import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 导入核心模块 (请确保在此脚本运行的根目录下能找到 core 包)
from core.robot_env import RobotEnv
from core.communication import CommunicationChannel
from core.controller import RobotController

# ==========================================
# 强化学习环境包装器: 将 JCC 转换为标准 Gym 环境
# ==========================================

class JCCRobotEnv(gym.Env):
    """
    联合通信与控制 (JCC) 的强化学习自定义环境。
    AI (Agent) 在这里扮演“智能带宽调度器”的角色，
    目标是用有限的 16 bits 带宽，使得机械臂的轨迹追踪误差（控制代价）最小。
    """
    
    def __init__(self, gui=False, B_total=16, threshold=0.08, max_steps=480):
        super(JCCRobotEnv, self).__init__()
        
        # 1. 核心物理与控制参数
        self.B_total = B_total
        self.threshold = threshold
        self.max_steps = max_steps
        self.current_step = 0
        self.dt = 1./240.
        
        # 2. 实例化三大核心模块
        self.env = RobotEnv(gui=gui, dt=self.dt)
        # 我们有4个状态变量: q_0, q_1, dq_0, dq_1
        self.channel = CommunicationChannel(num_dims=4) 
        self.controller = RobotController(Kp=[200.0, 200.0], Kd=[20.0, 20.0], Ki=[50.0, 50.0])
        
        # 3. 定义动作空间 (Action Space)
        # AI 输出一个 4 维连续向量 [-1, 1]，分别对应4个状态变量的分配权重
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # 4. 定义状态观测空间 (Observation Space)
        # 观测 8 维数据: [q0, q1, dq0, dq1, target_q0, target_q1, error_q0, error_q1]
        # 机械臂角度通常在 [-pi, pi] 之间，这里给一个较宽的安全范围 [-10, 10]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(8,), dtype=np.float32)

    def _get_obs(self):
        """
        获取当前环境的观测状态 (State) 喂给 AI
        """
        q_real, dq_real = self.env.get_true_state()
        error_q = q_real - self.q_target
        
        # 拼接 8 维观测向量
        obs = np.concatenate([q_real, dq_real, self.q_target, error_q])
        return obs.astype(np.float32)

    def _action_to_bits(self, action):
        """
        将 AI 的连续输出 [-1, 1]^4 映射为整数比特分配 [b1, b2, b3, b4]
        """
        # 1. 使用 Softmax 将动作转化为概率分布
        exp_a = np.exp(action)
        probs = exp_a / np.sum(exp_a)
        
        # 2. 按比例分配总带宽
        b_float = probs * self.B_total
        b_int = np.round(b_float).astype(int)
        
        # 3. 修复取整带来的总数误差，确保严格等于 B_total
        diff = int(self.B_total - np.sum(b_int))
        if diff > 0:
            # 少的比特，补给小数部分最大的那个
            for _ in range(diff):
                idx = np.argmax(b_float - b_int)
                b_int[idx] += 1
        elif diff < 0:
            # 多的比特，从分得最多的里面扣
            for _ in range(-diff):
                idx = np.argmax(b_int)
                b_int[idx] -= 1
                
        # 兜底：防止出现负数
        b_int = np.clip(b_int, 0, self.B_total)
        return b_int

    def reset(self, seed=None, options=None):
        """
        每个 Episode 开始时重置环境
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        # 重置物理环境和控制器积分项
        self.env.reset()
        self.controller.reset_integral()
        
        # 重置通信信道记忆
        self.channel.x_last_sent = np.zeros(4)
        
        # 设定初始目标 (t=0)
        self.q_target = np.array([0.0, -0.5])
        self.dq_target = np.array([0.0, 0.0])
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """
        环境推演核心逻辑：接收 AI 的比特分配 -> 通信压缩 -> 物理控制 -> 计算 Reward
        """
        t = self.current_step * self.dt
        
        # 1. 生成当前的物理控制目标
        self.q_target = np.array([0.5 * np.sin(2.0 * t), 0.5 * np.cos(2.0 * t) - 0.5])
        self.dq_target = np.array([1.0 * np.cos(2.0 * t), -1.0 * np.sin(2.0 * t)])
        
        # 2. 获取真实的物理状态
        q_real, dq_real = self.env.get_true_state()
        x_real = np.concatenate((q_real, dq_real))
        
        # 3. AI 行动映射：得到当前步的带宽分配方案
        custom_bits = self._action_to_bits(action)
        
        # 4. 经过通信信道 (ETC + 动态量化)
        # 注意：这里会调用我们在 communication.py 中修改的 custom_bits 参数
        x_received, triggered, bits_used = self.channel.transmit(
            x_real, 
            use_etc=True, 
            threshold=self.threshold, 
            use_quantization=True, 
            B_total=self.B_total, 
            custom_bits=custom_bits
        )
        
        # 拆解接收到的状态
        q_hat, dq_hat = x_received[:2], x_received[2:]
        
        # 5. 控制与物理步进
        tau = self.controller.compute_torque(self.q_target, q_hat, self.dq_target, dq_hat)
        self.env.apply_torque(tau)
        
        # 6. 计算奖励 (Reward) - 极其关键
        # 重罚基座关节0 (权重1000)，轻罚肩部关节1 (权重1)，促使AI保护关节0
        error_q = q_real - self.q_target
        cost = 1000.0 * (error_q[0]**2) + 1.0 * (error_q[1]**2) 
        
        # 强化学习是追求 Reward 最大化，所以代价加负号
        reward = -cost
        
        # 7. 回合结束判定
        self.current_step += 1
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            terminated = True
            
        # 安全/截断机制：如果由于AI分配极差导致机械臂严重失稳，提前结束并给个巨大的负惩罚
        if cost > 50.0:
            reward -= 500.0
            truncated = True
            
        info = {"cost": cost, "triggered": triggered, "bits_used": bits_used, "allocated_bits": custom_bits}
        
        obs = self._get_obs()
        
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.env.close()

# 简易单元测试 (确保 Gym 环境符合规范)
if __name__ == "__main__":
    print("=== 测试强化学习环境 rl_env_wrapper ===")
    env = JCCRobotEnv(gui=False)
    obs, _ = env.reset()
    print(f"初始观测向量: {obs.round(3)}, 维度: {obs.shape}")
    
    # 给 AI 一个随机动作
    random_action = env.action_space.sample()
    print(f"AI 随机动作输出: {random_action.round(3)}")
    
    obs, reward, term, trunc, info = env.step(random_action)
    print(f"实际分配比特: {info['allocated_bits']}")
    print(f"获得即时奖励: {reward:.3f}")
    
    env.close()
    print("=== 测试通过！环境可用于 Stable Baselines3 ===")
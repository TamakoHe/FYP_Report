import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import copy

# ==========================================
# 0. 设定计算设备
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 物理环境 (加入随机初始化)
# ==========================================
class NetworkedControlEnv:
    def __init__(self, lambda_cost=10.0):
        self.A = np.array([[1.05, 0.05], [0.05, 0.90]])
        self.B = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.Q = np.array([[100.0, 0.0], [0.0, 1.0]])
        self.R = np.array([[0.1, 0.0], [0.0, 0.1]])
        self.K = np.array([[1.0, 0.05], [0.05, 0.9]]) 
        self.lambda_cost = lambda_cost

    def reset(self, train=True):
        if train:
            # 【终极修复】：边缘样本挖掘 (Hard Negative Mining)
            sign_x1 = np.random.choice([-1.0, 1.0])
            sign_x2 = np.random.choice([-1.0, 1.0])
            self.x = np.array([sign_x1 * np.random.uniform(3.0, 6.0), 
                               sign_x2 * np.random.uniform(3.0, 6.0)])
        else:
            # 测试时固定极端恶劣开局，以对比性能
            self.x = np.array([5.0, -5.0])
            
        self.x_hat = np.array([0.0, 0.0])
        self.u = np.array([0.0, 0.0])
        return self._get_state()

    def _get_state(self):
        # 【架构升级】：听取建议，AI 的视野不能仅仅是"误差"，必须包含"绝对位置"
        error = self.x - self.x_hat
        # 将估计位置 (x_hat) 和 误差 (error) 拼接，形成 4维 状态空间
        state_array = np.concatenate((self.x_hat, error))
        return torch.FloatTensor(state_array).to(device)

    def step(self, action):
        if action == 1:
            self.x_hat = self.x.copy()
        else:
            self.x_hat = self.A @ self.x_hat + self.B @ self.u
            
        self.u = -self.K @ self.x_hat
        noise = np.random.normal(0, 0.2, size=2)
        self.x = self.A @ self.x + self.B @ self.u + noise
        
        physical_cost = self.x.T @ self.Q @ self.x
        comm_cost = self.lambda_cost * action
        
        # Reward 缩放，防止 Q 值爆炸导致网络崩溃
        reward = -(physical_cost + comm_cost) / 100.0
        
        return self._get_state(), reward, False

# ==========================================
# 2. DQN 神经网络
# ==========================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # 因为输入维度增加了，稍微加宽一点隐藏层
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 强化学习训练循环 (加入 Target Network)
# ==========================================
def train_dqn():
    env = NetworkedControlEnv(lambda_cost=15.0)
    
    # 【架构升级】：输入状态维度从 2 提升到 4 (x_hat + error)
    model = DQN(state_dim=4, action_dim=2).to(device)
    
    target_model = copy.deepcopy(model).to(device)
    target_model.eval()
    
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    loss_fn = nn.MSELoss()
    
    memory = deque(maxlen=5000)
    batch_size = 64
    gamma = 0.95
    epsilon = 1.0
    epsilon_decay = 0.98 
    epsilon_min = 0.01
    episodes = 1000 
    steps_per_episode = 50
    
    print(f"开始训练 DQN-Pro 通信触发器... (当前设备: {device}, 状态维度: 4)")
    for ep in range(episodes):
        state = env.reset(train=True)
        total_reward = 0
        comm_count = 0
        
        for t in range(steps_per_episode):
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                with torch.no_grad():
                    q_values = model(state)
                    action = torch.argmax(q_values).item()
                    
            next_state, reward, _ = env.step(action)
            memory.append((state, action, reward, next_state))
            
            state = next_state
            total_reward += reward * 100.0 
            comm_count += action
            
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states = zip(*batch)
                
                states = torch.stack(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.stack(next_states).to(device)
                
                with torch.no_grad():
                    max_next_q = target_model(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q
                
                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = loss_fn(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        if (ep + 1) % 10 == 0:
            target_model.load_state_dict(model.state_dict())
            
        if (ep+1) % 50 == 0:
            print(f"回合: {ep+1}/{episodes} | 总奖励: {total_reward:.1f} | 发送次数: {comm_count}/{steps_per_episode}")
            
    return model, env

# ==========================================
# 4. 深度对比测试与可视化
# ==========================================
def run_simulation(method, env, noise_seq, model=None, interval=1):
    steps = len(noise_seq)
    x = np.array([5.0, -5.0]) 
    x_hat = np.array([0.0, 0.0])
    u = np.array([0.0, 0.0])
    
    traj_x1 = []
    costs = []
    actions = []
    
    for t in range(steps):
        if method == 'dqn':
            # 【架构升级】：测试时也必须组装 4维 状态
            error = x - x_hat
            state_array = np.concatenate((x_hat, error))
            state_tensor = torch.FloatTensor(state_array).to(device)
            with torch.no_grad():
                action = torch.argmax(model(state_tensor)).item()
        elif method == 'ideal':
            action = 1
        elif method == 'periodic':
            action = 1 if t % interval == 0 else 0
            
        actions.append(action)
        
        if action == 1:
            x_hat = x.copy()
        else:
            x_hat = env.A @ x_hat + env.B @ u
            
        u = -env.K @ x_hat
        x = env.A @ x + env.B @ u + noise_seq[t]
        
        cost = x.T @ env.Q @ x
        traj_x1.append(x[0])
        costs.append(cost)
        
    return traj_x1, costs, actions

def test_and_plot(model, env):
    model.eval()
    steps = 60
    np.random.seed(42)
    noise_seq = np.random.normal(0, 0.2, size=(steps, 2))
    
    traj_dqn, costs_dqn, act_dqn = run_simulation('dqn', env, noise_seq, model=model)
    comm_count = sum(act_dqn)
    
    interval = max(1, int(steps / max(1, comm_count)))
    traj_per, costs_per, act_per = run_simulation('periodic', env, noise_seq, interval=interval)
    
    traj_id, costs_id, act_id = run_simulation('ideal', env, noise_seq)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 2, 1]})
    
    ax1.plot(traj_id, label="Ideal (100% BW)", color='green', linestyle='--', linewidth=2)
    ax1.plot(traj_per, label=f"Periodic (Rate: ~{comm_count/steps*100:.0f}%)", color='red', linestyle='-.', alpha=0.7)
    ax1.plot(traj_dqn, label=f"DQN Smart (Rate: {comm_count/steps*100:.0f}%)", color='blue', marker='o', markersize=4)
    ax1.axhline(0, color='gray', linestyle=':')
    ax1.set_title("System Trajectory Comparison (Closer to 0 is better)")
    ax1.set_ylabel("State 1 Value")
    ax1.legend(loc="upper right")
    ax1.grid(True)
    
    cum_costs_id = np.cumsum(costs_id)
    cum_costs_per = np.cumsum(costs_per)
    cum_costs_dqn = np.cumsum(costs_dqn)
    
    ax2.plot(cum_costs_id, label="Ideal Cost (Lower Bound)", color='green', linestyle='--', linewidth=2)
    ax2.plot(cum_costs_per, label="Periodic Cost", color='red', linestyle='-.')
    ax2.plot(cum_costs_dqn, label="DQN Smart Cost", color='blue', linewidth=2)
    ax2.set_title("Cumulative Control Cost (Lower is better precision)")
    ax2.set_ylabel("Cumulative $x^T Q x$")
    ax2.legend(loc="upper left")
    ax2.grid(True)
    
    ax3.vlines(np.where(np.array(act_per)==1)[0], ymin=0, ymax=0.4, color='red', label='Periodic Transmit')
    ax3.vlines(np.where(np.array(act_dqn)==1)[0], ymin=0.6, ymax=1.0, color='blue', label='DQN Transmit')
    ax3.set_yticks([0.2, 0.8])
    ax3.set_yticklabels(['Periodic', 'DQN'])
    ax3.set_xlabel("Time Step (k)")
    ax3.set_title("Communication Strategy 'Barcode'")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model, eval_env = train_dqn()
    print("\n训练完成，正在生成多维度对比图表...")
    test_and_plot(trained_model, eval_env)
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from typing import Callable

# 导入自定义的 7-DOF 环境
from rl.rl_env_wrapper import JCCRobotEnv

# ==========================================
# 强化学习训练脚本 (高维极客优化版)
# 针对 14维动作空间 与 Softmax 比特分配 进行了算法级调优
# ==========================================

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    学习率线性衰减调度器。
    随着训练进行（progress_remaining 从 1 降到 0），学习率逐渐减小，
    帮助 PPO 在训练后期进行更精细的“微调 (Fine-tuning)”，超越 LQR。
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train():
    # 1. 包装环境
    # 提示：请务必确保你在 rl_env_wrapper.py 中已经把 reward 进行了缩放！(如 reward = -cost / 100.0)
    env = JCCRobotEnv(gui=False, num_dof=7, B_total=56, threshold=0.1, max_steps=480)
    
    log_dir = "./logs/ppo_jcc_7dof_optimized/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    # 2. 实例化强化的 PPO 模型
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        # 【优化1】：使用学习率衰减，从 3e-4 平滑降到 0
        learning_rate=linear_schedule(3e-4),
        
        n_steps=4096,
        
        # 【优化2】：增大 Batch Size (从 128 提升到 256)，在 14 维连续空间中提供更准确的梯度方向
        batch_size=256,       
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        
        # 【优化3 (极其关键)】：熵系数 (Entropy Coefficient)。
        # 默认是 0。设为 0.01 可以强迫 AI 在训练中保持“好奇心”，
        # 抵消 Softmax 带来的马太效应，探索更多稀奇古怪的带宽分配组合，跳出局部最优。
        ent_coef=0.01,        
        
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_jcc_7dof_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./logs/checkpoints_7dof_opt/",
        name_prefix="ppo_jcc_7dof_opt"
    )

    # 【优化4】：延长训练步数，14维空间的组合爆炸需要至少 150 万步以上的试错
    total_timesteps = 1500000
    print(f"🚀 开始执行优化版 PPO 训练，目标步数: {total_timesteps}...")
    print("💡 提示: 已经启用学习率线性衰减与熵正则化(探索机制)。")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        model_path = "ppo_jcc_7dof_dynamic_allocator"
        model.save(model_path)
        print(f"✅ 训练完成！优化后的高维模型已保存至: {model_path}.zip")
        
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断，正在尝试保存...")
        model.save("ppo_jcc_7dof_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    train()
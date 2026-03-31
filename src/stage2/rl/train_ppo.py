import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# 导入我们自定义的环境
from rl.rl_env_wrapper import JCCRobotEnv

# ==========================================
# 强化学习训练脚本: 训练 PPO 动态带宽调度器
# 职责: 实例化环境、配置 PPO 算法、开启大规模训练并保存模型
# ==========================================

def train():
    # 1. 创建并包装环境
    # 训练时关闭 GUI (gui=False) 以极大地提高物理仿真速度
    env = JCCRobotEnv(gui=False, B_total=16, threshold=0.08, max_steps=480)
    
    # 使用 Monitor 包装环境以便记录训练统计数据（如每回合的奖励和长度）
    log_dir = "./logs/ppo_jcc_results/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    # 2. 实例化 PPO 模型
    # 策略网络：使用 MlpPolicy (多层感知机)，适合处理这种小维度的状态向量
    # 学习率建议从 3e-4 开始，n_steps 控制每次更新前的经验采集量
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./ppo_jcc_tensorboard/"
    )

    # 3. 设置回调函数 (可选但推荐)
    # 每隔 50,000 步保存一个检查点，防止训练中断
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./logs/checkpoints/",
        name_prefix="ppo_jcc_model"
    )

    # 4. 开始训练
    # 建议至少训练 300,000 到 500,000 步以获得稳定的动态分配策略
    total_timesteps = 500000
    print(f"🚀 开始训练 PPO 智能体，目标步数: {total_timesteps}...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # 5. 保存最终训练好的“大脑”
        model_path = "ppo_jcc_dynamic_allocator"
        model.save(model_path)
        print(f"✅ 训练完成！模型已保存至: {model_path}.zip")
        
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断，正在尝试保存当前模型...")
        model.save("ppo_jcc_interrupted")
    finally:
        env.close()

if __name__ == "__main__":
    # 执行训练
    train()
    
    # 提示用户如何查看进度
    print("\n💡 提示: 您可以在终端输入以下命令查看训练曲线:")
    print("tensorboard --logdir ./ppo_jcc_tensorboard/")
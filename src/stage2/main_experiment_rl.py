import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# 导入核心模块
from core.robot_env import RobotEnv
from core.communication import CommunicationChannel
from core.controller import RobotController

# ==========================================
# 实验主程序: 强化学习 (RL) vs 传统策略 对比仿真
# ==========================================

def action_to_bits(action, B_total=16):
    """
    辅助函数：将 RL 输出的连续值映射为整数比特分配 (保持与 Wrapper 逻辑一致)
    """
    exp_a = np.exp(action)
    probs = exp_a / np.sum(exp_a)
    b_float = probs * B_total
    b_int = np.round(b_float).astype(int)
    
    # 修复余数，确保总和严格等于 B_total
    diff = int(B_total - np.sum(b_int))
    if diff > 0:
        for _ in range(diff):
            idx = np.argmax(b_float - b_int)
            b_int[idx] += 1
    elif diff < 0:
        for _ in range(-diff):
            idx = np.argmax(b_int)
            b_int[idx] -= 1
    return np.clip(b_int, 0, B_total)

def run_simulation(strategy_name, model=None, threshold=0.08, B_total=16):
    """
    运行单次实验仿真并收集指标
    """
    print(f"正在测试策略: [{strategy_name}] ...")
    env = RobotEnv(gui=False, dt=1./240.)
    channel = CommunicationChannel(num_dims=4)
    controller = RobotController(Kp=[200.0, 200.0], Kd=[20.0, 20.0], Ki=[50.0, 50.0])
    
    # 静态 LQR 权重计算 (用于策略对比)
    A_lin = np.eye(4) + np.diag([1/240., 1/240.], k=2)
    B_lin = np.vstack((np.zeros((2,2)), np.eye(2)))
    Q_lin = np.diag([1000.0, 1.0, 10.0, 0.1])
    R_lin = np.eye(2) * 0.1
    P_matrix, _ = RobotController.get_lqr_sensitivity(A_lin, B_lin, Q_lin, R_lin)
    lqr_weights = np.diag(P_matrix) if P_matrix is not None else np.ones(4)
    
    total_time = 4.0
    steps = int(total_time / env.dt)
    
    # 记录器
    log_q_real = []
    log_q_target = []
    log_transmissions = []
    total_bits = 0
    
    env.reset()
    controller.reset_integral()
    
    for step in range(steps):
        t = step * env.dt
        q_target = np.array([0.5 * np.sin(2.0 * t), 0.5 * np.cos(2.0 * t) - 0.5])
        dq_target = np.array([1.0 * np.cos(2.0 * t), -1.0 * np.sin(2.0 * t)])
        
        q_real, dq_real = env.get_true_state()
        x_real = np.concatenate((q_real, dq_real))
        
        custom_bits = None
        
        # 策略逻辑分支
        if strategy_name == 'Ideal':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=False, use_quantization=False)
        elif strategy_name == 'ETC_Only':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=False)
        elif strategy_name == 'Uniform_ETC':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=True, B_total=B_total)
        elif strategy_name == 'LQR_Weighted_ETC':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=True, weights=lqr_weights, variances=np.ones(4)*0.1, B_total=B_total)
        elif strategy_name == 'PPO_RL_Dynamic':
            # 1. 构造 AI 观测向量 (8维)
            error_q = q_real - q_target
            obs = np.concatenate([q_real, dq_real, q_target, error_q]).astype(np.float32)
            # 2. AI 预测决策
            action, _ = model.predict(obs, deterministic=True)
            custom_bits = action_to_bits(action, B_total)
            # 3. 通信执行
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=True, B_total=B_total, custom_bits=custom_bits)
            
        q_hat, dq_hat = x_recv[:2], x_recv[2:]
        tau = controller.compute_torque(q_target, q_hat, dq_target, dq_hat)
        env.apply_torque(tau)
        
        # 记录
        log_q_real.append(q_real.copy())
        log_q_target.append(q_target.copy())
        log_transmissions.append(1 if triggered else 0)
        total_bits += bits
        
    env.close()
    
    # ==========================================
    # 指标计算 (定量衡量控制与通信效果)
    # ==========================================
    log_q_real = np.array(log_q_real)
    log_q_target = np.array(log_q_target)
    
    # 计算误差
    error_q = log_q_real - log_q_target
    
    # 1. 均方误差 (MSE) 和 均方根误差 (RMSE)
    mse_q0 = np.mean(error_q[:, 0]**2)
    mse_q1 = np.mean(error_q[:, 1]**2)
    rmse_q0 = np.sqrt(mse_q0)
    rmse_q1 = np.sqrt(mse_q1)
    
    # 2. 最大绝对误差 (Max Error - 衡量极限安全性)
    max_err_q0 = np.max(np.abs(error_q[:, 0]))
    max_err_q1 = np.max(np.abs(error_q[:, 1]))
    
    # 3. 综合加权控制代价 (基于Q矩阵)
    weighted_mse = 1000.0 * mse_q0 + 1.0 * mse_q1
    
    # 4. 通信指标
    transmissions_count = int(np.sum(log_transmissions))
    transmission_rate = (transmissions_count / steps) * 100.0
    
    print(f"  📊 [通信指标] 触发率: {transmission_rate:.1f}% ({transmissions_count}/{steps}次), 消耗总带宽: {total_bits} bits")
    print(f"  📉 [整体代价] 综合加权代价 (Weighted MSE): {weighted_mse:.4f}")
    print(f"  🎯 [各关节详情]")
    print(f"      - 关节 0 (基座-极敏感): RMSE = {rmse_q0:.4f} rad, 最大偏差 = {max_err_q0:.4f} rad")
    print(f"      - 关节 1 (肩部-低敏感): RMSE = {rmse_q1:.4f} rad, 最大偏差 = {max_err_q1:.4f} rad\n")
    
    return {
        'q_real': log_q_real, 'q_target': log_q_target,
        'transmissions': np.array(log_transmissions),
        'total_bits': total_bits, 'cost': weighted_mse,
        'rmse_q0': rmse_q0, 'rmse_q1': rmse_q1,
        'max_err_q0': max_err_q0, 'max_err_q1': max_err_q1,
        'transmission_rate': transmission_rate
    }

def plot_final_comparison(results):
    """
    绘制五路策略对比图
    """
    time_axis = np.linspace(0, 4.0, len(results['Ideal']['q_real']))
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 2, 1]})
    
    styles = {
        'Ideal': ('green', '--', 'Ideal (100% BW)'),
        'ETC_Only': ('gray', ':', 'ETC Only'),
        'Uniform_ETC': ('red', '-.', 'Uniform Quant'),
        'LQR_Weighted_ETC': ('blue', '-', 'LQR Weighted'),
        'PPO_RL_Dynamic': ('purple', '-', 'PPO Dynamic (AI)')
    }

    # 1. 关节 0 (高敏感)
    axes[0].plot(time_axis, results['Ideal']['q_target'][:, 0], 'k--', alpha=0.3, label='Target')
    for name, (color, ls, label) in styles.items():
        axes[0].plot(time_axis, results[name]['q_real'][:, 0], color=color, linestyle=ls, 
                     label=f"{label} [Cost: {results[name]['cost']:.2f}]")
    axes[0].set_title('High-Sensitivity Joint (Base) - Comparison')
    axes[0].legend(loc='upper right', fontsize='small')
    axes[0].grid(True)

    # 2. 关节 1 (低敏感)
    axes[1].plot(time_axis, results['Ideal']['q_target'][:, 1], 'k--', alpha=0.3)
    for name, (color, ls, label) in styles.items():
        axes[1].plot(time_axis, results[name]['q_real'][:, 1], color=color, linestyle=ls)
    axes[1].set_title('Low-Sensitivity Joint (Shoulder) - Comparison')
    axes[1].grid(True)

    # 3. 通信资源对比
    names = list(results.keys())
    total_bits_list = [results[n]['total_bits'] for n in names]
    axes[2].barh(names, total_bits_list, color=[styles[n][0] for n in names])
    axes[2].set_title('Total Communication Bandwidth Consumed (Bits)')
    axes[2].set_xlabel('Total Bits')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 加载 RL 模型
    try:
        ppo_model = PPO.load("ppo_jcc_dynamic_allocator")
        print("✅ 成功加载训练好的 PPO 模型。")
    except:
        print("❌ 未找到模型文件，请先运行 rl/train_ppo.py 进行训练！")
        exit()

    # 运行五种策略对比
    all_res = {}
    all_res['Ideal'] = run_simulation('Ideal')
    all_res['ETC_Only'] = run_simulation('ETC_Only', threshold=0.08)
    all_res['Uniform_ETC'] = run_simulation('Uniform_ETC', threshold=0.08, B_total=16)
    all_res['LQR_Weighted_ETC'] = run_simulation('LQR_Weighted_ETC', threshold=0.08, B_total=16)
    all_res['PPO_RL_Dynamic'] = run_simulation('PPO_RL_Dynamic', model=ppo_model, threshold=0.08, B_total=16)

    # 绘图
    plot_final_comparison(all_res)
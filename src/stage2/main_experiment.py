import numpy as np
import matplotlib.pyplot as plt
import time

# 导入我们封装好的三大核心模块
from core.robot_env import RobotEnv
from core.communication import CommunicationChannel
from core.controller import RobotController

# ==========================================
# 实验主程序: 联合通信与控制 (JCC) 仿真对比
# ==========================================

def run_simulation(strategy_name, use_gui=False, threshold=0.1, B_total=16, use_weighted=False):
    """
    运行单次仿真实验
    """
    print(f"\n🚀 开始仿真策略: [{strategy_name}]")
    
    # 1. 初始化模块
    env = RobotEnv(gui=use_gui, dt=1./240.)
    # 状态维度为 4 (2个角度 q + 2个角速度 dq)
    channel = CommunicationChannel(num_dims=4)
    controller = RobotController(Kp=[200.0, 200.0], Kd=[20.0, 20.0], Ki=[50.0, 50.0])
    
    # 2. 获取 LQR 敏感度矩阵 P (用于加权压缩)
    # 构造一个简化的 4D 线性化系统来提取敏感度
    A_lin = np.eye(4) + np.diag([1/240., 1/240.], k=2)
    B_lin = np.vstack((np.zeros((2,2)), np.eye(2)))
    # Q矩阵：极度厌恶关节0(基座)的位置误差，权重给到1000
    Q_lin = np.diag([1000.0, 1.0, 10.0, 0.1]) 
    R_lin = np.eye(2) * 0.1
    
    P_matrix, _ = RobotController.get_lqr_sensitivity(A_lin, B_lin, Q_lin, R_lin)
    lqr_weights = np.diag(P_matrix) if P_matrix is not None else np.ones(4)
    variances = np.ones(4) * 0.1 # 假设传感器噪声方差
    
    # 3. 仿真循环参数
    total_time = 4.0
    steps = int(total_time / env.dt)
    
    # 数据记录器
    log_q_real = np.zeros((steps, 2))
    log_q_target = np.zeros((steps, 2))
    log_transmissions = np.zeros(steps)
    total_bits = 0
    
    env.reset()
    controller.reset_integral()
    
    # 4. 步进循环
    for step in range(steps):
        t = step * env.dt
        
        # [目标生成]
        q_target = np.array([0.5 * np.sin(2.0 * t), 0.5 * np.cos(2.0 * t) - 0.5])
        dq_target = np.array([1.0 * np.cos(2.0 * t), -1.0 * np.sin(2.0 * t)])
        
        # [环境层]: 获取真实物理状态
        q_real, dq_real = env.get_true_state()
        x_real = np.concatenate((q_real, dq_real))
        
        # [通信层]: 核心策略分支
        if strategy_name == 'Ideal':
            # 理想情况：不触发ETC，不压缩，全量发送
            x_received, triggered, bits = channel.transmit(
                x_real, use_etc=False, use_quantization=False
            )
        elif strategy_name == 'ETC_Only':
            # 仅事件触发，不压缩
            x_received, triggered, bits = channel.transmit(
                x_real, use_etc=True, threshold=threshold, use_quantization=False
            )
        elif strategy_name == 'Uniform_ETC':
            # 传统方案：均匀压缩 + ETC
            x_received, triggered, bits = channel.transmit(
                x_real, use_etc=True, threshold=threshold, 
                use_quantization=True, weights=np.ones(4), variances=variances, B_total=B_total
            )
        elif strategy_name == 'LQR_Weighted_ETC':
            # 创新方案：LQR加权压缩 + ETC
            x_received, triggered, bits = channel.transmit(
                x_real, use_etc=True, threshold=threshold, 
                use_quantization=True, weights=lqr_weights, variances=variances, B_total=B_total
            )
            
        # 拆解接收到的状态
        q_hat, dq_hat = x_received[:2], x_received[2:]
        
        # [控制层]: 计算力矩
        tau = controller.compute_torque(q_target, q_hat, dq_target, dq_hat)
        
        # [环境层]: 施加力矩
        env.apply_torque(tau)
        
        # 数据记录
        log_q_real[step] = q_real
        log_q_target[step] = q_target
        log_transmissions[step] = 1 if triggered else 0
        total_bits += bits
        
    env.close()
    
    # 简单计算 MSE 作为物理代价的参考
    mse_q0 = np.mean((log_q_real[:, 0] - log_q_target[:, 0])**2)
    mse_q1 = np.mean((log_q_real[:, 1] - log_q_target[:, 1])**2)
    weighted_mse = 1000.0 * mse_q0 + 1.0 * mse_q1 # 使用Q矩阵的权重来评估
    
    print(f"📊 结果: 传输 {int(np.sum(log_transmissions))}/{steps} 次, 消耗总带宽: {total_bits} bits")
    print(f"📉 加权控制代价 (Weighted MSE): {weighted_mse:.4f}")
    
    return {
        'q_real': log_q_real,
        'q_target': log_q_target,
        'transmissions': log_transmissions,
        'total_bits': total_bits,
        'cost': weighted_mse
    }

# ==========================================
# 绘制综合对比大图
# ==========================================
def plot_results(results_dict, dt=1./240.):
    steps = len(list(results_dict.values())[0]['q_real'])
    time_axis = np.arange(steps) * dt
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 2, 1]})
    colors = {'Ideal': 'green', 'ETC_Only': 'gray', 'Uniform_ETC': 'red', 'LQR_Weighted_ETC': 'blue'}
    line_styles = {'Ideal': '--', 'ETC_Only': ':', 'Uniform_ETC': '-.', 'LQR_Weighted_ETC': '-'}
    
    # 1. 关节 0 轨迹 (高敏感度)
    axes[0].plot(time_axis, list(results_dict.values())[0]['q_target'][:, 0], 'k--', label='Target', alpha=0.5)
    for name, res in results_dict.items():
        axes[0].plot(time_axis, res['q_real'][:, 0], color=colors[name], linestyle=line_styles[name], label=f'{name} (Cost: {res["cost"]:.2f})')
    axes[0].set_title('Joint 0 Trajectory (Highly Sensitive Base Joint)')
    axes[0].set_ylabel('Angle (rad)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. 关节 1 轨迹 (低敏感度)
    axes[1].plot(time_axis, list(results_dict.values())[0]['q_target'][:, 1], 'k--', label='Target', alpha=0.5)
    for name, res in results_dict.items():
        axes[1].plot(time_axis, res['q_real'][:, 1], color=colors[name], linestyle=line_styles[name], label=name)
    axes[1].set_title('Joint 1 Trajectory (Low Sensitive Shoulder Joint)')
    axes[1].set_ylabel('Angle (rad)')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. 通信条形码
    for idx, (name, res) in enumerate(results_dict.items()):
        if name == 'Ideal': continue # 理想状态全满，不画条形码以免干扰
        transmissions = np.where(res['transmissions'] == 1)[0] * dt
        axes[2].vlines(transmissions, ymin=idx*0.3, ymax=idx*0.3+0.2, color=colors[name], label=f'{name} (Bits: {res["total_bits"]})')
    
    axes[2].set_title('Communication Barcode (Transmission Events)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_yticks([])
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 为了加快跑多组对比数据的速度，这里 use_gui=False。你可以设为 True 观看最后一次机械臂动画
    res_ideal = run_simulation('Ideal', use_gui=False)
    res_etc = run_simulation('ETC_Only', use_gui=False, threshold=0.08)
    
    # 设定极度严苛的带宽：每次通信只允许发 16 bits (原来4个浮点数需要 256 bits!)
    res_uniform = run_simulation('Uniform_ETC', use_gui=False, threshold=0.08, B_total=16)
    res_weighted = run_simulation('LQR_Weighted_ETC', use_gui=False, threshold=0.08, B_total=16)
    
    # 组装结果并画图
    all_results = {
        'Ideal': res_ideal,
        'ETC_Only': res_etc,
        'Uniform_ETC': res_uniform,
        'LQR_Weighted_ETC': res_weighted
    }
    plot_results(all_results)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import pybullet as p
import pybullet_data

# 导入核心物理与控制模块
from core.robot_env import RobotEnv
from core.communication import CommunicationChannel
from core.controller import RobotController

# ==========================================
# Phase 1: 纯软件物理仿真实验 - 数据采集、评估与可视化脚本 (7-DOF)
# 稳定版：真实的 Pick-and-Place (抓取与放置) 物理任务验证
# ==========================================

# 设置数据保存路径
SAVE_DIR = "results/phase1_sim_7dof"
os.makedirs(SAVE_DIR, exist_ok=True)

def action_to_bits(action, B_total=56, min_bits=2):
    """【终极安全版】：保底 2 bits + 动态分配剩余带宽，杜绝 0-bit 死亡陷阱"""
    n = len(action) # 14维状态变量
    dynamic_budget = B_total - n * min_bits # 56 - 28 = 28 bits
    
    # 适度的温度缩放，帮助拉开分配差距
    temp = 3.0
    scaled_a = action * temp
    # 减去最大值防溢出 (NaN保护)
    exp_a = np.exp(scaled_a - np.max(scaled_a))
    probs = exp_a / np.sum(exp_a)
    
    b_dynamic = probs * dynamic_budget
    b_int = np.floor(b_dynamic).astype(int) # 向下取整保证不超标
    
    # 将向下取整损失的余数，补偿给分配比例最高的状态
    diff = int(dynamic_budget - np.sum(b_int))
    for _ in range(diff):
        b_int[np.argmax(b_dynamic - b_int)] += 1
        
    final_bits = b_int + min_bits
    return np.clip(final_bits, 0, B_total)

def run_single_strategy(strategy_name, model=None, threshold=0.1, B_total=56, num_dof=7):
    """运行单次 7-DOF 策略，执行抓取与放置任务，返回时序记录字典和定量指标"""
    print(f"🔄 正在运行 7-DOF 抓取与放置仿真: [{strategy_name}] ...")
    
    # 提示：如果想亲眼观看机械臂抓取物体的过程，请将 gui=False 改为 gui=True
    env = RobotEnv(gui=False, dt=1./240.)
    
    # 自动解锁至 7 自由度
    if hasattr(env, 'controlled_joints') and len(env.controlled_joints) < num_dof:
        env.controlled_joints = list(range(num_dof))
        for j in env.controlled_joints:
            p.setJointMotorControl2(env.robotId, j, p.VELOCITY_CONTROL, force=0)

    # ---------------------------------------------------
    # 加载物理抓取对象与目标区域，并生成逆运动学轨迹
    # ---------------------------------------------------
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 加载一个 2.0kg 的重物作为负载扰动！(颜色：橙色)
    obj_id = p.loadURDF("cube_small.urdf", [0.5, 0.0, 0.025], globalScaling=1.2)
    p.changeDynamics(obj_id, -1, mass=2.0) 
    p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0.5, 0, 1]) 
    
    # 加载一个目标放置区域指示器 (颜色：半透明绿色)
    target_id = p.loadURDF("cube_small.urdf", [0.0, 0.5, 0.025], globalScaling=1.5, useFixedBase=True)
    p.changeVisualShape(target_id, -1, rgbaColor=[0, 1, 0, 0.3])
    
    # 利用 PyBullet 逆运动学(IK)预先生成关键帧的关节角度
    q_home = np.zeros(num_dof)
    ee_orientation = p.getQuaternionFromEuler([0, np.pi, 0]) # 末端执行器垂直朝下
    
    def get_ik(pos):
        return np.array(p.calculateInverseKinematics(env.robotId, 6, pos, ee_orientation))[:num_dof]
        
    q_hover_pick = get_ik([0.5, 0.0, 0.3])   # 在重物正上方
    q_pick       = get_ik([0.5, 0.0, 0.12])  # 贴近重物准备抓取
    q_hover_place= get_ik([0.0, 0.5, 0.3])   # 移动到目标区域正上方
    q_place      = get_ik([0.0, 0.5, 0.12])  # 放下重物
    
    # 定义样条关键帧：(时间, 关节角度)
    keyframes = [
        (0.0, q_home),
        (1.0, q_hover_pick),
        (1.5, q_pick),
        (2.0, q_pick),         # 停顿 0.5秒 以便抓取
        (2.5, q_hover_pick),
        (3.5, q_hover_place),
        (4.0, q_place),
        (4.5, q_place),        # 停顿 0.5秒 以便释放
        (5.0, q_hover_place)
    ]
    
    def get_target(t):
        """利用余弦插值生成极其平滑且速度连续的参考轨迹"""
        for i in range(len(keyframes)-1):
            t0, q0 = keyframes[i]
            t1, q1 = keyframes[i+1]
            if t0 <= t <= t1:
                if t1 == t0: return q1, np.zeros(num_dof)
                phase = (t - t0) / (t1 - t0)
                s = 0.5 - 0.5 * np.cos(phase * np.pi)
                q = q0 + (q1 - q0) * s
                ds = 0.5 * np.pi * np.sin(phase * np.pi) / (t1 - t0)
                dq = (q1 - q0) * ds
                return q, dq
        return keyframes[-1][1], np.zeros(num_dof)

    # ---------------------------------------------------

    channel = CommunicationChannel(num_dims=num_dof * 2)
    controller = RobotController(Kp=[200.0]*num_dof, Kd=[20.0]*num_dof, Ki=[50.0]*num_dof)
    
    A_lin = np.eye(num_dof * 2) + np.diag([1/240.]*num_dof, k=num_dof)
    B_lin = np.vstack((np.zeros((num_dof, num_dof)), np.eye(num_dof)))
    
    Q_diag = np.concatenate([
        np.array([1000.0, 800.0, 500.0, 200.0, 50.0, 10.0, 1.0]), 
        np.ones(num_dof) * 0.1
    ])
    Q_lin = np.diag(Q_diag)
    R_lin = np.eye(num_dof) * 0.1
    
    P_matrix, _ = RobotController.get_lqr_sensitivity(A_lin, B_lin, Q_lin, R_lin)
    lqr_weights = np.diag(P_matrix) if P_matrix is not None else np.ones(num_dof * 2)
    
    total_time = 5.0 
    steps = int(total_time / env.dt)
    
    history = {
        'time': np.zeros(steps),
        'triggered': np.zeros(steps, dtype=int),
        'bits_used': np.zeros(steps, dtype=int)
    }
    for i in range(num_dof):
        history[f'q{i}_real'] = np.zeros(steps)
        history[f'q{i}_target'] = np.zeros(steps)
    
    total_bits = 0
    env.reset()
    controller.reset_integral()
    channel.x_last_sent = np.zeros(num_dof * 2)
    grasp_constraint = None 
    
    for step in range(steps):
        t = step * env.dt
        
        q_target, dq_target = get_target(t)
        
        # 物理抓取与释放逻辑
        if 1.5 <= t < 4.5 and grasp_constraint is None:
            grasp_constraint = p.createConstraint(
                parentBodyUniqueId=env.robotId, parentLinkIndex=6,
                childBodyUniqueId=obj_id, childLinkIndex=-1,
                jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0.05], childFramePosition=[0, 0, 0]
            )
        elif t >= 4.5 and grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
            grasp_constraint = None
            
        q_real_full, dq_real_full = env.get_true_state()
        
        if len(q_real_full) < num_dof:
            q_real = np.pad(q_real_full, (0, num_dof - len(q_real_full)), 'constant')
            dq_real = np.pad(dq_real_full, (0, num_dof - len(dq_real_full)), 'constant')
        else:
            q_real = q_real_full[:num_dof]
            dq_real = dq_real_full[:num_dof]
            
        x_real = np.concatenate((q_real, dq_real))
        custom_bits = None
        
        if strategy_name == 'Ideal':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=False, use_quantization=False)
        elif strategy_name == 'ETC_Only':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=False)
        elif strategy_name == 'Uniform_ETC':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=True, B_total=B_total)
        elif strategy_name == 'LQR_Weighted_ETC':
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=True, weights=lqr_weights, variances=np.ones(num_dof*2)*0.1, B_total=B_total)
        elif strategy_name == 'PPO_RL_Dynamic':
            error_q = q_real - q_target
            obs = np.concatenate([q_real, dq_real, q_target, error_q]).astype(np.float32)
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
                custom_bits = action_to_bits(action, B_total)
            else:
                custom_bits = np.ones(num_dof * 2) * (B_total // (num_dof * 2))
            x_recv, triggered, bits = channel.transmit(x_real, use_etc=True, threshold=threshold, use_quantization=True, B_total=B_total, custom_bits=custom_bits)
            
        q_hat, dq_hat = x_recv[:num_dof], x_recv[num_dof:]
        tau_active = controller.compute_torque(q_target, q_hat, dq_target, dq_hat)
        tau_command = tau_active[:len(q_real_full)]
        env.apply_torque(tau_command)
        
        history['time'][step] = t
        history['triggered'][step] = 1 if triggered else 0
        history['bits_used'][step] = bits
        total_bits += bits
        for i in range(num_dof):
            history[f'q{i}_real'][step] = q_real[i]
            history[f'q{i}_target'][step] = q_target[i]
        
    env.close()
    
    metrics = {
        'Strategy': strategy_name,
        'Total_Bits': total_bits,
        'Transmission_Rate (%)': (np.sum(history['triggered']) / steps) * 100.0
    }
    
    weighted_mse = 0
    for i in range(num_dof):
        error_qi = history[f'q{i}_real'] - history[f'q{i}_target']
        metrics[f'RMSE_J{i}'] = np.sqrt(np.mean(error_qi**2))
        metrics[f'Max_Err_J{i}'] = np.max(np.abs(error_qi))
        weighted_mse += Q_diag[i] * np.mean(error_qi**2)
        
    metrics['Weighted_MSE_Cost'] = weighted_mse
    
    return history, metrics

def plot_trajectories(all_hist_data, df_metrics):
    """绘制轨迹对比图并保存，直接可用作论文插图"""
    print("📊 正在生成并保存抓取与放置任务的轨迹对比图...")
    
    time_axis = all_hist_data['Ideal']['time']
    
    joints_to_plot = [0, 3, 6]
    joint_names = ['Joint 0 (Base - High Sensitivity)', 'Joint 3 (Middle)', 'Joint 6 (End-effector - Low Sensitivity)']
    
    strategies_to_plot = ['Uniform_ETC', 'LQR_Weighted_ETC', 'PPO_RL_Dynamic']
    styles = {
        'Uniform_ETC': {'color': 'red', 'ls': '-.', 'label': 'Uniform Allocation'},
        'LQR_Weighted_ETC': {'color': 'blue', 'ls': ':', 'label': 'LQR Static'},
        'PPO_RL_Dynamic': {'color': 'purple', 'ls': '-', 'label': 'PPO Dynamic (Ours)'}
    }
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 2, 2, 1.5]})
    
    for idx, (j_idx, j_name) in enumerate(zip(joints_to_plot, joint_names)):
        ax = axes[idx]
        
        target_traj = all_hist_data['Ideal'][f'q{j_idx}_target']
        ax.plot(time_axis, target_traj, 'k--', linewidth=2, label='Target Trajectory', alpha=0.7)
        
        for strat in strategies_to_plot:
            if strat in all_hist_data:
                real_traj = all_hist_data[strat][f'q{j_idx}_real']
                ax.plot(time_axis, real_traj, color=styles[strat]['color'], 
                        linestyle=styles[strat]['ls'], label=styles[strat]['label'], linewidth=1.5)
        
        ax.axvspan(1.5, 4.5, color='orange', alpha=0.1, label='Payload Zone (2.0kg)')
        
        ax.set_title(j_name, fontsize=14, fontweight='bold')
        ax.set_ylabel('Angle (rad)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
            
    ax_bar = axes[3]
    costs = [df_metrics[df_metrics['Strategy'] == s]['Weighted_MSE_Cost'].values[0] for s in strategies_to_plot]
    labels = [styles[s]['label'] for s in strategies_to_plot]
    colors = [styles[s]['color'] for s in strategies_to_plot]
    
    bars = ax_bar.bar(labels, costs, color=colors, alpha=0.8, width=0.5)
    ax_bar.set_title('Overall Control Cost Under Payload Disturbance (Lower is Better)', fontsize=14, fontweight='bold')
    ax_bar.set_ylabel('Weighted MSE Cost', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2, yval + (max(costs)*0.02), 
                    f'{yval:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    
    plot_path = os.path.join(SAVE_DIR, "trajectory_comparison_pick_place.png")
    plt.savefig(plot_path, dpi=300)
    print(f"✅ 高清图表已保存至: {plot_path}")
    
    plt.show()

def collect_phase1_data():
    """统筹运行所有基线并保存数据与图片"""
    print("="*60)
    print("🚀 启动 Phase 1 抓取与放置(Pick-and-Place) 验证任务...")
    print("="*60)
    
    try:
        ppo_model = PPO.load("ppo_jcc_7dof_dynamic_allocator")
        print("✅ 检测到 PPO_7DOF 预训练模型。")
    except Exception as e:
        print("⚠️ 未找到 7-DOF PPO 模型，将回退为均匀分配！请确保已运行 rl/train_ppo.py。")
        ppo_model = None

    strategies = ['Ideal', 'ETC_Only', 'Uniform_ETC', 'LQR_Weighted_ETC', 'PPO_RL_Dynamic']
    
    all_metrics = []
    all_hist_data = {} 
    
    for strat in strategies:
        hist_data, metric_data = run_single_strategy(strat, model=ppo_model, threshold=0.1, B_total=56, num_dof=7)
        all_metrics.append(metric_data)
        all_hist_data[strat] = hist_data
        
        df_hist = pd.DataFrame(hist_data)
        csv_path = os.path.join(SAVE_DIR, f"trajectory_{strat}.csv")
        df_hist.to_csv(csv_path, index=False)
        print(f"   💾 {strat} 时序数据已保存至 CSV")

    df_metrics = pd.DataFrame(all_metrics)
    report_path = os.path.join(SAVE_DIR, "metrics_summary.csv")
    df_metrics.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print("✅ 物理抓取验证完成，准备绘图！")
    
    plot_trajectories(all_hist_data, df_metrics)
    
    print("="*60)

if __name__ == "__main__":
    collect_phase1_data()
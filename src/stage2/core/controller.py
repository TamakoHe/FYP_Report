import numpy as np
import scipy.linalg as la

# ==========================================
# 核心模块: 控制层 (Controller Layer)
# 职责: 1. 基于接收到的状态 (可能被压缩或延迟) 计算控制力矩
#       2. 提供 LQR 敏感度矩阵给通信层用于加权压缩
# ==========================================

class RobotController:
    def __init__(self, Kp, Kd, Ki=None, dt=1./240., integral_limit=2.0):
        """
        初始化机械臂 PID 控制器
        Args:
            Kp (np.ndarray): 比例增益 (刚度)
            Kd (np.ndarray): 微分增益 (阻尼)
            Ki (np.ndarray, optional): 积分增益 (用于消除重力造成的稳态误差)
            dt (float): 控制周期
            integral_limit (float): 积分限幅，防止积分饱和 (Anti-windup)
        """
        self.Kp = np.array(Kp)
        self.Kd = np.array(Kd)
        self.Ki = np.array(Ki) if Ki is not None else np.zeros_like(self.Kp)
        self.dt = dt
        self.integral_limit = integral_limit
        
        # 积分器状态
        self.error_integral = np.zeros_like(self.Kp)

    def reset_integral(self):
        """
        重置积分器 (在每次新的仿真回合开始时调用)
        """
        self.error_integral = np.zeros_like(self.Kp)

    def compute_torque(self, q_target, q_hat, dq_target, dq_hat):
        """
        核心控制逻辑：基于接收到的状态 (q_hat, dq_hat) 计算下发给电机的力矩。
        注意: 控制器不知道真实的 q_real，它只能相信通信层传过来的 q_hat。
        
        Args:
            q_target (np.ndarray): 目标关节角度
            q_hat (np.ndarray): 接收到的当前关节角度估算值
            dq_target (np.ndarray): 目标关节角速度
            dq_hat (np.ndarray): 接收到的当前关节角速度估算值
            
        Returns:
            tau (np.ndarray): 计算出的控制力矩
        """
        # 1. 计算误差 (基于接收到的数据)
        error_q = q_target - q_hat
        error_dq = dq_target - dq_hat
        
        # 2. 积分项累加与限幅 (Anti-windup)
        self.error_integral += error_q * self.dt
        self.error_integral = np.clip(self.error_integral, -self.integral_limit, self.integral_limit)
        
        # 3. 计算 PID 力矩
        tau = self.Kp * error_q + self.Kd * error_dq + self.Ki * self.error_integral
        
        return tau

    @staticmethod
    def get_lqr_sensitivity(A, B, Q, R):
        """
        求解离散时间代数 Riccati 方程 (DARE)，提取敏感度矩阵 P
        该矩阵将提供给通信层，用于指导"基于控制关键性"的非均匀比特分配。
        
        Args:
            A, B: 系统线性化动力学矩阵
            Q, R: LQR 状态与控制惩罚矩阵
            
        Returns:
            P (np.ndarray): 敏感度矩阵 (Hessian Matrix)
            K (np.ndarray): LQR 最优反馈增益矩阵 (可选备用)
        """
        try:
            # 求解 DARE 得到矩阵 P
            P = la.solve_discrete_are(A, B, Q, R)
            # 计算 LQR 的最优反馈增益 K = (R + B^T P B)^-1 B^T P A
            K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
            return P, K
        except Exception as e:
            print(f"[Controller Error] 无法求解 Riccati 方程，请检查矩阵维度和系统可控性: {e}")
            return None, None


# ==========================================
# 单元测试代码 (仅在此文件被直接运行时执行)
# ==========================================
if __name__ == "__main__":
    print("=== 开始单元测试: core/controller.py ===")
    
    # --- 测试 1: PID 力矩计算 ---
    print("\n[测试 1] PID 力矩计算功能:")
    controller = RobotController(Kp=[100.0, 100.0], Kd=[10.0, 10.0], Ki=[50.0, 50.0])
    
    q_tgt = np.array([1.0, 0.5])
    dq_tgt = np.array([0.0, 0.0])
    
    # 假设由于通信延迟或压缩，控制器收到的 q_hat 是有偏差的
    q_hat_received = np.array([0.9, 0.4]) 
    dq_hat_received = np.array([0.0, 0.0])
    
    tau = controller.compute_torque(q_tgt, q_hat_received, dq_tgt, dq_hat_received)
    print(f"  目标位置: {q_tgt}")
    print(f"  接收位置: {q_hat_received}")
    print(f"  计算力矩: {tau} (应主要由 Kp * 0.1 产生，约等于 10.0)")
    
    # --- 测试 2: LQR 敏感度矩阵计算 ---
    print("\n[测试 2] LQR 敏感度矩阵 P 的计算:")
    # 构造一个简易的 2维 LTI 系统
    A_test = np.array([[1.05, 0.05], [0.05, 0.90]])
    B_test = np.array([[1.0, 0.0], [0.0, 1.0]])
    Q_test = np.array([[1000.0, 0.0], [0.0, 1.0]]) # 状态1极其敏感
    R_test = np.array([[0.1, 0.0], [0.0, 0.1]])
    
    P_matrix, K_gain = RobotController.get_lqr_sensitivity(A_test, B_test, Q_test, R_test)
    
    if P_matrix is not None:
        weights = np.diag(P_matrix)
        print(f"  成功求解 Riccati 方程！")
        print(f"  敏感度矩阵 P 的对角线元素 (Weights): {weights.round(2)}")
        print(f"  预期结果: 第一个元素应该远大于第二个元素，因为 Q 矩阵中状态 1 的惩罚极大。")
    
    print("\n=== 单元测试结束 ===")
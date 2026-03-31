import pybullet as p
import pybullet_data
import time
import numpy as np

# ==========================================
# 核心模块: 物理引擎层 (Robot Environment)
# 职责: 封装 PyBullet，负责机械臂加载、状态读取和力矩执行
# ==========================================

class RobotEnv:
    def __init__(self, gui=True, dt=1./240.):
        """
        初始化物理环境、加载机械臂、设置仿真参数
        """
        self.dt = dt
        # 连接物理引擎
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        
        # 加载环境与机械臂
        self.planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        # 使用 KUKA iiwa 作为测试床
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation, useFixedBase=True)
        
        # 我们只控制前两个关节 (Base and Shoulder) 作为 2-DOF 测试对象
        self.controlled_joints = [0, 1]
        numJoints = p.getNumJoints(self.robotId)
        
        # 锁死其他不需要控制的关节，防止干扰
        for j in range(numJoints):
            if j not in self.controlled_joints:
                p.setJointMotorControl2(self.robotId, j, p.POSITION_CONTROL, targetPosition=0, force=500)
                
        # 禁用受控关节的默认速度/位置控制，准备接收我们在控制层计算的力矩 (Torque Control)
        for j in self.controlled_joints:
            p.setJointMotorControl2(self.robotId, j, p.VELOCITY_CONTROL, force=0)
            
    def reset(self):
        """
        重置机械臂状态到初始零位
        """
        for j in self.controlled_joints:
            p.resetJointState(self.robotId, j, targetValue=0.0, targetVelocity=0.0)
        return self.get_true_state()

    def get_true_state(self):
        """
        模拟传感器: 读取关节的真实角度 q 和角速度 dq
        Returns:
            q_real (np.ndarray): 当前关节角度
            dq_real (np.ndarray): 当前关节角速度
        """
        joint_states = p.getJointStates(self.robotId, self.controlled_joints)
        q_real = np.array([state[0] for state in joint_states])
        dq_real = np.array([state[1] for state in joint_states])
        return q_real, dq_real

    def apply_torque(self, tau):
        """
        执行器: 接收控制器算出来的力矩，下发给电机，并推演一个物理步
        Args:
            tau (np.ndarray): 长度为2的力矩数组 [tau_0, tau_1]
        """
        p.setJointMotorControlArray(
            bodyUniqueId=self.robotId,
            jointIndices=self.controlled_joints,
            controlMode=p.TORQUE_CONTROL,
            forces=tau
        )
        # 物理引擎演化一步
        p.stepSimulation()
        
    def close(self):
        """
        关闭物理引擎连接
        """
        p.disconnect(self.physicsClient)


# ==========================================
# 单元测试代码 (仅在此文件被直接运行时执行)
# ==========================================
if __name__ == "__main__":
    print("=== 开始单元测试: core/robot_env.py ===")
    
    # 1. 实例化环境 (打开 GUI)
    env = RobotEnv(gui=True)
    env.reset()
    
    # 2. 设置 PID 控制器 (加入积分项以克服重力造成的稳态误差)
    Kp = np.array([200.0, 200.0]) # 适当调大比例增益
    Kd = np.array([20.0, 20.0])   # 适当调大阻尼
    Ki = np.array([50.0, 50.0])   # 新增: 积分增益，用于消除重力误差
    
    error_integral = np.zeros(2)  # 累计误差容器
    
    print("机械臂将在 PID控制下移动到目标位置 [0.5, -0.5] rad...")
    
    try:
        # 运行 4 秒钟的仿真 (240 Hz * 4 = 960 steps)
        for step in range(240 * 4):
            # 获取真实状态
            q_real, dq_real = env.get_true_state()
            
            # 设定一个静态目标位置
            q_target = np.array([0.5, -0.5])
            dq_target = np.array([0.0, 0.0])
            
            # 计算误差
            error_q = q_target - q_real
            error_dq = dq_target - dq_real
            
            # 积分项累加 (误差 * dt)
            error_integral += error_q * env.dt
            # 为了安全，限制一下积分项的最大值 (防积分饱和 Anti-windup)
            error_integral = np.clip(error_integral, -2.0, 2.0)
            
            # 计算 PID 测试力矩
            tau = Kp * error_q + Kd * error_dq + Ki * error_integral
            
            # 施加力矩并步进
            env.apply_torque(tau)
            
            # 减速以便肉眼观察
            time.sleep(env.dt)
            
            # 每隔1秒打印一次状态
            if step % 240 == 0:
                print(f"[Time: {step/240:.1f}s] 当前角度 q = {q_real.round(3)}, 误差 = {error_q.round(3)}, 下发力矩 tau = {tau.round(3)}")
                
    finally:
        print("单元测试结束，正在关闭物理环境。")
        env.close()
import pybullet as p
import pybullet_data
import time
import math

# ==========================================
# 1. 初始化 PyBullet 物理引擎
# ==========================================
# 连接到GUI界面
physicsClient = p.connect(p.GUI)
# 设置资源搜索路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# 设置重力
p.setGravity(0, 0, -9.81)

# 配置视角以便更好地观察
p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.7, 0, 0])

# ==========================================
# 2. 加载环境与模型
# ==========================================
# 加载地面
planeId = p.loadURDF("plane.urdf")

# 定义位置A和位置B
POS_A = [0.7, -0.2, 0.025]  
POS_B = [0.7, 0.2, 0.025]   

# 加载小方块
cubeId = p.loadURDF("cube_small.urdf", basePosition=POS_A)

# 加载 Franka Panda 7自由度机械臂
robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)

# ==========================================
# 3. 定义控制函数
# ==========================================
# Franka Panda 的末端执行器(End-Effector)所在的 link 索引
EE_LINK_INDEX = 11

# ==========================================
# 3.1 定义网络和执行器类 (有限带宽: 8-bit 精度量化模拟)
# ==========================================
class ActuatorGroup:
    def __init__(self, robot_id, actuator_ids):
        self.robot_id = robot_id
        self.current_targets = {}
        # 初始化时获取当前关节位置作为初始目标
        for aid in actuator_ids:
            joint_state = p.getJointState(self.robot_id, aid)
            self.current_targets[aid] = joint_state[0]

    def update_setpoint(self, actuator_id, target):
        """接收来自网络的最新目标点（带有量化误差）"""
        self.current_targets[actuator_id] = target

    def step(self):
        """执行器本地高频控制循环：向目标点驱动"""
        for aid, target in self.current_targets.items():
            force = 50.0 if aid in [9, 10] else 500.0
            p.setJointMotorControl2(
                bodyIndex=self.robot_id, jointIndex=aid, 
                controlMode=p.POSITION_CONTROL, targetPosition=target, 
                force=force, maxVelocity=2.0
            )

class NetworkSimulator:
    def __init__(self, robot_id, actuator_ids, bits_per_joint=4):
        self.robot_id = robot_id
        self.actuator_ids = actuator_ids
        self.bits_per_joint = bits_per_joint
        self.max_int = (1 << bits_per_joint) - 1  # 例如 8-bit 的最大整数是 255
        
        self.joint_limits = {}
        # 动态获取每个关节的物理限位 (lower_limit, upper_limit)
        for aid in actuator_ids:
            info = p.getJointInfo(self.robot_id, aid)
            lower_limit = info[8]
            upper_limit = info[9]
            # 容错处理：如果没有明确限位，赋予一个默认合理区间
            if lower_limit >= upper_limit:
                if aid in [9, 10]: # 夹爪
                    lower_limit, upper_limit = 0.0, 0.04
                else:
                    lower_limit, upper_limit = -2.0 * math.pi, 2.0 * math.pi
            self.joint_limits[aid] = (lower_limit, upper_limit)
            
        self.buffers = {aid: None for aid in actuator_ids} 

    def send(self, actuator_id, target):
        """
        控制器将指令发到网络：在这里模拟有限精度带宽。
        包含：浮点数 -> N-bit 整数 (传输) -> 浮点数 (带有精度损失)
        """
        lower, upper = self.joint_limits[actuator_id]
        
        # 1. 限制目标在物理范围内
        target_clamped = max(lower, min(target, upper))
        
        # 2. 编码 (Encode): 归一化后映射到 [0, max_int] 的离散整数
        normalized = (target_clamped - lower) / (upper - lower)
        quantized_int = int(round(normalized * self.max_int))
        quantized_int = max(0, min(quantized_int, self.max_int))  # 确保整数不越界
        
        # --- (此处代表网络传输了该 quantized_int 数据包) ---
        
        # 3. 解码 (Decode): 执行器接收到的整数还原回物理位置
        decoded_target = lower + (quantized_int / self.max_int) * (upper - lower)
        
        self.buffers[actuator_id] = decoded_target

    def step(self, actuators: ActuatorGroup):
        """每一个控制周期，将数据透明传输给执行器（高频但低精度）"""
        for aid in self.actuator_ids:
            if self.buffers[aid] is not None:
                actuators.update_setpoint(aid, self.buffers[aid])

# 实例化架构体系
ACTUATOR_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 10] # 7个关节 + 2个夹爪

# 【核心修改点】每个自由度被分配 8 bit (即总线在单次下发时载荷为 56+16 bit)
BITS_PER_JOINT = 4
actuators = ActuatorGroup(robotId, ACTUATOR_IDS)
network = NetworkSimulator(robotId, ACTUATOR_IDS, bits_per_joint=BITS_PER_JOINT)

def move_robot_ee(target_pos, target_orn, duration=2.0):
    """
    移动机械臂末端到指定的位置和姿态
    """
    steps = int(duration * 240)  
    for _ in range(steps):
        # [1. 控制器端] 计算逆运动学（IK），这是一个高精度的 float 数值
        joint_poses = p.calculateInverseKinematics(
            robotId, EE_LINK_INDEX, target_pos, target_orn, 
            maxNumIterations=100, residualThreshold=1e-5
        )
        
        # [2. 控制器端发包] 经过网络发送，精度被强制截断到 8-bit
        for i in range(7):
            network.send(actuator_id=i, target=joint_poses[i])
            
        # [3. 网络传输] 无延迟，但带着量化误差送达
        network.step(actuators=actuators)
        
        # [4. 执行器端] 电机只能追踪那些“跳跃”的低精度位置指令
        actuators.step()
        
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

def control_gripper(target_width, duration=1.0):
    """
    控制夹爪开合 (当前仅作视觉展示，物理抓取由约束接管)
    """
    steps = int(duration * 240)
    for _ in range(steps):
        # 夹爪同样经过 8-bit 带宽限制网络
        network.send(actuator_id=9, target=target_width)
        network.send(actuator_id=10, target=target_width)
        network.step(actuators=actuators)
        actuators.step()
        
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

# ==========================================
# 4. 执行状态机流程
# ==========================================
print("仿真开始...")

# 预设姿态：末端执行器垂直朝下
down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])

# 各个关键点的位置
hover_z = 0.25      
grasp_z = 0.04     

hover_pos_A = [POS_A[0], POS_A[1], hover_z]
grasp_pos_A = [POS_A[0], POS_A[1], grasp_z]

hover_pos_B = [POS_B[0], POS_B[1], hover_z]
grasp_pos_B = [POS_B[0], POS_B[1], grasp_z]

# 流程 1: 张开夹爪
print("-> 初始化夹爪张开")
control_gripper(target_width=0.04, duration=0.5)

# 流程 2: 移动到位置A的正上方
print("-> 移动到物体 A 正上方")
move_robot_ee(hover_pos_A, down_orientation, duration=1.0)

# 流程 3: 下降到物体A位置
print("-> 下降准备抓取")
move_robot_ee(grasp_pos_A, down_orientation, duration=0.5)

# 流程 4: 激活粘连约束并闭合夹爪
print("-> 激活固定约束 (Sticky Grasp)")
control_gripper(target_width=0.01, duration=0.2)
grasp_constraint = p.createConstraint(
    parentBodyUniqueId=robotId,
    parentLinkIndex=EE_LINK_INDEX,
    childBodyUniqueId=cubeId,
    childLinkIndex=-1,
    jointType=p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0],
    childFramePosition=[0, 0, 0]
)

# 缓冲一小会
for _ in range(60):  
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

# 流程 5: 抬起物体
print("-> 抬起物体")
move_robot_ee(hover_pos_A, down_orientation, duration=0.5)

# 流程 6: 移动到目标位置B的正上方
print("-> 移动到位置 B 正上方")
move_robot_ee(hover_pos_B, down_orientation, duration=1.0)

# 流程 7: 下降到位置B
print("-> 下降放置物体")
move_robot_ee(grasp_pos_B, down_orientation, duration=0.5)

# 流程 8: 解除约束释放物体
print("-> 解除约束并释放物体")
p.removeConstraint(grasp_constraint)
control_gripper(target_width=0.04, duration=0.2)

# 流程 9: 机械臂复位
print("-> 任务完成，机械臂抬起复位")
move_robot_ee(hover_pos_B, down_orientation, duration=0.5)

print("仿真动作结束，将在 5 秒后自动关闭...")

# ==========================================
# 5. 自动退出机制
# ==========================================
for _ in range(240 * 5):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()
print("仿真已自动断开并安全退出。")
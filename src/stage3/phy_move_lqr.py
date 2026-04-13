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
    def __init__(self, robot_id, actuator_ids):
        self.robot_id = robot_id
        self.actuator_ids = actuator_ids
        
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

    def allocate_bits(self, sensitivities, joint_ranges, total_bits=56, min_bits=4, max_bits=16):
        """
        根据信息论的率失真理论进行最优比特分配。
        增加 min_bits 到 4，防止极低精度导致的严重非线性畸变。
        """
        n = len(sensitivities)
        
        # 1. 计算每个关节的信息重要性权重 (Jacobian * 物理范围)
        weights = [sensitivities[i] * joint_ranges[i] for i in range(n)]
        weights = [max(w, 1e-6) for w in weights] # 防止 log(0)
        
        # 2. 迭代注水算法 (Water-filling) 以满足 min_bits 与 max_bits 约束
        bits_float = [0.0] * n
        unfixed = list(range(n))
        
        while True:
            if not unfixed:
                break
            
            sum_log2 = sum(math.log2(weights[i]) for i in unfixed)
            # 计算基准线 (Offset)
            offset = (total_bits - sum(bits_float[j] for j in range(n) if j not in unfixed) - sum_log2) / len(unfixed)
            
            newly_fixed = False
            for i in unfixed:
                b = offset + math.log2(weights[i])
                if b < min_bits:
                    bits_float[i] = min_bits
                    unfixed.remove(i)
                    newly_fixed = True
                    break
                elif b > max_bits:
                    bits_float[i] = max_bits
                    unfixed.remove(i)
                    newly_fixed = True
                    break
                    
            if not newly_fixed:
                for i in unfixed:
                    bits_float[i] = offset + math.log2(weights[i])
                break

        # 3. 小数部分离散化为整数，并保证总和精确等于 total_bits
        bits_int = [int(math.floor(b)) for b in bits_float]
        remainder = total_bits - sum(bits_int)
        
        # 优先把多余的 1 bit 补给小数部分被砍掉最多的关节
        frac_parts = [(bits_float[i] - bits_int[i], i) for i in range(n)]
        frac_parts.sort(reverse=True, key=lambda x: x[0])
        
        for i in range(int(remainder)):
            idx = frac_parts[i][1]
            bits_int[idx] += 1
            
        return bits_int

    def quantize_value(self, actuator_id, target, bits):
        """核心量化逻辑"""
        lower, upper = self.joint_limits[actuator_id]
        target_clamped = max(lower, min(target, upper))
        
        max_int = (1 << bits) - 1
        if max_int <= 0:
            return target_clamped
        
        normalized = (target_clamped - lower) / (upper - lower)
        quantized_int = int(round(normalized * max_int))
        quantized_int = max(0, min(quantized_int, max_int))
        
        return lower + (quantized_int / max_int) * (upper - lower)

    def send(self, actuator_id, target, bits=8):
        """控制器将指令发到网络"""
        decoded_target = self.quantize_value(actuator_id, target, bits)
        self.buffers[actuator_id] = decoded_target

    def step(self, actuators: ActuatorGroup):
        """每一个控制周期，将数据透明传输给执行器"""
        for aid in self.actuator_ids:
            if self.buffers[aid] is not None:
                actuators.update_setpoint(aid, self.buffers[aid])

# 实例化架构体系
ACTUATOR_IDS = [0, 1, 2, 3, 4, 5, 6, 9, 10]

# ==========================================
# 【统一带宽参数】总带宽，方便统一做实验 
# (建议 >= 28，因为 min_bits=4 且有 7个关节)
# ==========================================
TOTAL_ARM_BITS = 14

# 计算平均分配方案的基础位宽 (尽可能均匀分配 TOTAL_ARM_BITS 给 7 个关节)
AVG_ALLOCATED_BITS = [TOTAL_ARM_BITS // 7] * 7
for i in range(TOTAL_ARM_BITS % 7):
    AVG_ALLOCATED_BITS[i] += 1

actuators = ActuatorGroup(robotId, ACTUATOR_IDS)
network = NetworkSimulator(robotId, ACTUATOR_IDS)

# ==========================================
# 3.2 定义量化评估指标记录器
# ==========================================
eval_metrics = {
    'steps': 0,
    'error_avg_allocation': 0.0,
    'error_dyn_allocation': 0.0
}

# 引入防抖机制：每 24 帧（约 0.1s）重新分配一次带宽
ALLOCATION_INTERVAL = 24
# 初始分配使用平均分配计算结果
current_allocated_bits = list(AVG_ALLOCATED_BITS)  

def move_robot_ee(target_pos, target_orn, duration=2.0):
    """
    移动机械臂末端到指定的位置和姿态
    """
    global eval_metrics, current_allocated_bits
    steps = int(duration * 240)  
    
    for step_count in range(steps):
        # [1. 控制器端] 计算逆运动学（IK）
        joint_poses = p.calculateInverseKinematics(
            robotId, EE_LINK_INDEX, target_pos, target_orn, 
            maxNumIterations=100, residualThreshold=1e-5
        )
        
        # [2 & 3. 动态敏感性计算与带宽分配 - 仅在特定间隔执行以防抖]
        if step_count % ALLOCATION_INTERVAL == 0:
            joint_states = p.getJointStates(robotId, range(7))
            q = [state[0] for state in joint_states]
            q_padded = q + [0.0, 0.0] 
            zero_vec = [0.0] * 9
            
            J_t, J_r = p.calculateJacobian(robotId, EE_LINK_INDEX, [0,0,0], q_padded, zero_vec, zero_vec)
            
            sensitivities = []
            joint_ranges = []
            for i in range(7):
                # 【优化点】：引入姿态雅可比。0.2 为一个经验性的末端力臂权重
                s_t = J_t[0][i]**2 + J_t[1][i]**2 + J_t[2][i]**2
                s_r = J_r[0][i]**2 + J_r[1][i]**2 + J_r[2][i]**2
                s = math.sqrt(s_t + 0.2 * s_r)
                sensitivities.append(s)
                
                lower, upper = network.joint_limits[i]
                joint_ranges.append(upper - lower)
                
            # 动态分配计算
            current_allocated_bits = network.allocate_bits(
                sensitivities, joint_ranges, total_bits=TOTAL_ARM_BITS, min_bits=4
            )

        # 为了评估误差，我们需要当前帧的雅可比平移矩阵
        if step_count % ALLOCATION_INTERVAL != 0:
             joint_states = p.getJointStates(robotId, range(7))
             q = [state[0] for state in joint_states]
             q_padded = q + [0.0, 0.0] 
             zero_vec = [0.0] * 9
             J_t, _ = p.calculateJacobian(robotId, EE_LINK_INDEX, [0,0,0], q_padded, zero_vec, zero_vec)

        # ========================================================
        # [评估逻辑] 量化对比：平均分配 VS 动态分配(LQR式)
        # ========================================================
        dq_avg = []
        dq_dyn = []
        for i in range(7):
            target_q = joint_poses[i]
            # 模拟平均分配下的解码值 (利用统一计算的平均位宽参数)
            q_avg = network.quantize_value(i, target_q, bits=AVG_ALLOCATED_BITS[i])
            q_dyn = network.quantize_value(i, target_q, bits=current_allocated_bits[i])
            
            dq_avg.append(q_avg - target_q)
            dq_dyn.append(q_dyn - target_q)
            
        err_avg_sq = 0.0
        err_dyn_sq = 0.0
        for dim in range(3): 
            val_avg = sum(J_t[dim][i] * dq_avg[i] for i in range(7))
            val_dyn = sum(J_t[dim][i] * dq_dyn[i] for i in range(7))
            err_avg_sq += val_avg**2
            err_dyn_sq += val_dyn**2
            
        eval_metrics['error_avg_allocation'] += math.sqrt(err_avg_sq)
        eval_metrics['error_dyn_allocation'] += math.sqrt(err_dyn_sq)
        eval_metrics['steps'] += 1
        # ========================================================
        
        # [4. 控制器端发包] 经过网络发送，各关节按各自位宽进行截断
        for i in range(7):
            network.send(actuator_id=i, target=joint_poses[i], bits=current_allocated_bits[i])
            
        # [5. 网络传输] 
        network.step(actuators=actuators)
        
        # [6. 执行器端] 
        actuators.step()
        
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

def control_gripper(target_width, duration=1.0):
    """控制夹爪开合"""
    steps = int(duration * 240)
    for _ in range(steps):
        network.send(actuator_id=9, target=target_width, bits=8)
        network.send(actuator_id=10, target=target_width, bits=8)
        network.step(actuators=actuators)
        actuators.step()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

# ==========================================
# 4. 执行状态机流程
# ==========================================
print("仿真开始...")

down_orientation = p.getQuaternionFromEuler([math.pi, 0, 0])

hover_z = 0.25      
grasp_z = 0.04     

hover_pos_A = [POS_A[0], POS_A[1], hover_z]
grasp_pos_A = [POS_A[0], POS_A[1], grasp_z]

hover_pos_B = [POS_B[0], POS_B[1], hover_z]
grasp_pos_B = [POS_B[0], POS_B[1], grasp_z]

print("-> 初始化夹爪张开")
control_gripper(target_width=0.04, duration=0.5)

print("-> 移动到物体 A 正上方")
move_robot_ee(hover_pos_A, down_orientation, duration=1.0)

print("-> 下降准备抓取")
move_robot_ee(grasp_pos_A, down_orientation, duration=0.5)

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

for _ in range(60):  
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

print("-> 抬起物体")
move_robot_ee(hover_pos_A, down_orientation, duration=0.5)

print("-> 移动到位置 B 正上方")
move_robot_ee(hover_pos_B, down_orientation, duration=1.0)

print("-> 下降放置物体")
move_robot_ee(grasp_pos_B, down_orientation, duration=0.5)

print("-> 解除约束并释放物体")
p.removeConstraint(grasp_constraint)
control_gripper(target_width=0.04, duration=0.2)

print("-> 任务完成，机械臂抬起复位")
move_robot_ee(hover_pos_B, down_orientation, duration=0.5)

# ==========================================
# 5. 输出量化评估结果
# ==========================================
print("\n" + "="*45)
print(f" 🚀 网络带宽分配方案量化评估报告 ({TOTAL_ARM_BITS} Bit 总线)")
print("="*45)
mean_err_avg = eval_metrics['error_avg_allocation'] / eval_metrics['steps']
mean_err_dyn = eval_metrics['error_dyn_allocation'] / eval_metrics['steps']
improvement = (mean_err_avg - mean_err_dyn) / mean_err_avg * 100

print(f"1. 平均分配方案 (近似 {TOTAL_ARM_BITS/7:.1f}-bit/DOF) 末端平均绝对误差 : {mean_err_avg * 1000:.3f} mm")
print(f"2. 动态分配方案 (雅可比感知分配)末端平均绝对误差 : {mean_err_dyn * 1000:.3f} mm")
print("-" * 45)
if improvement > 0:
    print(f"✅ 结论: 动态分配使机械臂末端追踪精度提升了 {improvement:.1f} %")
else:
    print(f"⚠️ 结论: 在此轨迹下动态分配无明显优势 ({improvement:.1f} %)")
print("="*45 + "\n")

print("仿真动作结束，将在 5 秒后自动关闭...")

# ==========================================
# 6. 自动退出机制
# ==========================================
for _ in range(240 * 5):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()
print("仿真已自动断开并安全退出。")
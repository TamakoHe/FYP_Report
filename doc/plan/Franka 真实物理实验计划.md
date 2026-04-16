# **Real-World Franka Emika Experimental Plan (Sim-to-Real)**

## **1\. 硬件架构与通信仿真层设计 (Hardware & Software Architecture)**

在真实的 Franka 控制系统中，Franka Control Interface (FCI) 强制要求以 **1 kHz (1000 Hz)** 的频率持续发送平滑的控制指令。如果我们直接把 10 Hz 且经过极端量化的“阶跃信号”喂给机械臂，会立刻触发保护机制。

因此，真实实验的系统架构必须包含以下三个核心节点：

1. **JCC 编码端 (Sensor & Allocator)**: 运行我们训练好的 DRL 模型。读取 Franka 当前状态，以 **10 Hz** 的频率计算 ![][image1]，进行 DPCM 量化和位宽分配。
2. **虚拟受限信道 (Constrained Network Emulator)**: 在 ROS/Python 内部人为截断数据，强制将数据包压缩至 ![][image2] (例如 28-bit)。
3. **解码与高频平滑层 (Decoder & 1kHz Interpolator)**: 接收 10Hz 的量化航点。使用 **样条插值 (Spline Interpolation)** 或 **低通滤波器 (Low-Pass Filter)**，将 10Hz 的粗糙阶跃信号平滑上采样至 1000Hz，再下发给 FCI 执行。

## **2\. 核心物理实验设计 (Core Experimental Scenarios)**

### **Experiment A: 零样本真实世界迁移 (Zero-Shot Sim-to-Real Transfer)**

* **实验目的**: 证明在 PyBullet 理想环境中训练出的 DRL 模型，能够直接（Zero-shot）部署在真实的 Franka 机械臂上，并克服真实世界中未建模的摩擦力（Friction）和传感器噪声（Sensor Noise）。
* **实验步骤**:
  1. 设定安全带宽约束：**28-bit**（在真实硬件上，14-bit 的物理抖动可能导致关节过热，我们使用 28-bit 作为基准测试带宽，平均每关节 4 bits）。
  2. 让真实的 Franka 执行与仿真中相同的 3D Pick-and-Place 轨迹。
  3. 对比 **Uniform (Average)**, **Static LQR**, 和 **DRL** 在真实抓取点和放置点的物理稳态误差（可通过前向运动学读取，或通过外部相机标定）。
* **学术亮点**: 在真实世界中，由于静摩擦力和电机死区的存在，Uniform 分配会导致末端产生明显的“肉眼可见的微小抖动”。DRL 通过牺牲手腕精度，能让大臂表现出极其丝滑的运动轨迹。

### **Experiment B: 动态负载扰动测试 (Dynamic Payload Robustness) —— 【杀手锏实验】**

* **实验目的**: 这是彻底宣判 Static LQR 死刑，证明 DRL 具备“自适应生命力”的最强实验。
* **物理痛点**: Static LQR 的敏感度矩阵是在机械臂**空载 (0 kg)** 的情况下计算的。
* **实验步骤**:
  1. 保持带宽为 28-bit。
  2. 在 Franka 的夹爪上固定一个 **1.0 kg 或 2.0 kg 的重物**（模拟抓取到了重型工件）。
  3. 再次运行三种算法。
* **预期结果与学术亮点**:
  加入重物后，机械臂的惯性张量 ![][image3] 发生剧变。**Static LQR 会彻底崩溃**，因为它还在按照空载的逻辑分配 bits，导致承受巨大重力的大臂关节因为缺乏 bit 而疯狂掉高度。
  **相反，DRL 模型从未在带有负载的环境中训练过！** 但是，DRL 的 34 维状态空间中包含 ![][image4] 的实时追踪误差。当大臂因为拿了重物而掉高度时，DRL 会立刻“感受”到这个误差的剧增，并**本能地将手腕的 bit 抢过来，补偿给大臂**，从而在未知负载下依然稳稳到达目标点！这证明了 DRL 具有超越数学公式的**隐式自适应鲁棒性 (Implicit Adaptive Robustness)**。

### **Experiment C: 硬件级抖动与能耗分析 (Hardware Jitter & Energy Analysis)**

* **实验目的**: 从工业应用的角度，评估不同算法对硬件寿命的保护作用。
* **实验步骤**:
  1. 提取法兰卡末端关节（如 Joint 6 和 Joint 7）在运行过程中的**实时角速度 (![][image5]) 和 指令扭矩 ()** 曲线。
  2. 计算命令扭矩的变化率的平方积分（即 Jerk/Energy 消耗）。
* **学术亮点**: DRL 策略因为在末端分配了极低的 bit（比如 1 bit），可能会在理论上造成末端速度跳变。但由于我们加入的高频平滑层，真实扭矩会被过滤得很平滑。这个实验用来向评委证明：尽管 DRL 进行了极端的“偏科”压缩，但在硬件执行层面是**绝对安全且平滑**的。

## **3\. 实验数据采集清单与图表规划 (Data Collection & Plotting)**

在进行真实实验时，请务必记录以下数据流 (ROS bag 或 CSV 格式)，用于在论文中生成精美的图表：

1. **Fig X.1: 真实世界轨迹对比 (Real-world EE Trajectory)** \* 采集 Franka 末端执行器的实际笛卡尔坐标 ![][image7] 随时间变化曲线。
2. **Table X.1: 真实稳态误差表 (Real-world Steady-State Errors)**
   * 记录在 21-bit 和 28-bit 约束下，空载与满载 (1.0kg) 时的 Grasp Error 和 Place Error 毫米数。
3. **Fig X.2: 扭矩/速度抖动对比图 (Joint Velocity/Torque Jitter)**
   * 抽取一段运动过程，对比 Uniform 和 DRL 的关节 1（底座）和关节 7（手腕）的真实电机速度波动。证明 DRL “稳住了底座”。
4. **Fig X.3: 真实实验延时摄影 (Time-lapse Photography)**
   * 架设一个三脚架，用相机拍摄 Franka 执行任务的连拍照片（或者视频截图拼接）。图注上标明：“DRL successfully grasps the target under 28-bit constraint with a 1kg payload.” 这种具有视觉冲击力的实物图是工科顶级论文的标配。

## **4\. 实施阶段与排期 (Implementation Timeline)**

* **Phase 1 (Day 1-2)**: 开发 10Hz \-\> 1000Hz 的高频平滑插值节点（ZOH \+ Spline Interpolator），并在不限制位宽的情况下，确保 Franka 能平滑跟随。
* **Phase 2 (Day 3-4)**: 接入 JCC 逻辑，强制执行 28-bit 截断，跑通 Uniform 和 Static LQR 基线。
* **Phase 3 (Day 5-6)**: 载入仿真中训练好的 PPO 模型 (ppo\_multicore\_physical\_allocator.zip)，执行 Zero-shot 测试并采集无负载数据。
* **Phase 4 (Day 7\)**: 挂载重物（1.0kg 砝码），采集 Payload Robustness 的震撼数据。

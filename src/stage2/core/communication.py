import numpy as np

# ==========================================
# 核心模块: 通信与压缩层 (Communication Layer)
# 职责: 模拟网络信道，执行事件触发 (ETC) 和量化压缩 (支持静态LQR和动态RL分配)
# ==========================================

class CommunicationChannel:
    def __init__(self, num_dims):
        """
        初始化通信信道
        Args:
            num_dims (int): 状态变量的维度
        """
        self.num_dims = num_dims
        # 接收端/控制器的记忆容器 (用于 Zero-Order Hold 零阶保持)
        self.x_last_sent = np.zeros(num_dims)
        
        # 统计数据
        self.total_transmissions = 0
        self.total_bits_used = 0

    def check_event_trigger(self, x_current, threshold):
        """
        判断是否需要发包 (事件触发机制)
        如果当前状态与上次发送状态的误差范数大于阈值，则触发。
        """
        error = np.linalg.norm(x_current - self.x_last_sent)
        return error > threshold

    def _allocate_bits(self, weights, variances, B_total):
        """
        内部方法：基于率失真理论的 LQR 加权比特分配算法 (静态分配)
        """
        n = len(weights)
        b = np.zeros(n)
        
        # 计算几何平均值
        product_term = np.prod(weights * variances)
        # 防止 product_term 为 0 导致错误
        if product_term <= 0:
            return np.full(n, B_total // n) # 降级为均匀分配
            
        geom_mean = product_term ** (1/n)
        
        for i in range(n):
            if weights[i] * variances[i] == 0:
                b[i] = 0
            else:
                b[i] = (B_total / n) + 0.5 * np.log2((weights[i] * variances[i]) / geom_mean)
                
        # 四舍五入，并约束在 [0, B_total] 内
        b = np.round(b)
        b = np.clip(b, 0, B_total)
        
        # 修复分配后总数不等于 B_total 的情况
        diff = int(B_total - np.sum(b))
        if diff > 0:
            b[np.argmax(weights)] += diff # 剩余的给最敏感的
        elif diff < 0:
            while diff < 0:
                idx = np.argmax(b)
                b[idx] -= 1
                diff += 1
                
        return b.astype(int)

    def lqr_weighted_quantize(self, x, weights, variances, B_total, val_range, custom_bits=None):
        """
        根据敏感度矩阵 P (对应weights) 或外部指定的 custom_bits 对数据进行非均匀量化截断
        """
        # 1. 确定比特分配方案
        if custom_bits is not None:
            # 强化学习动态传入
            bits_array = np.array(custom_bits, dtype=int)
        else:
            # LQR 静态自动计算
            bits_array = self._allocate_bits(weights, variances, B_total)
        
        # 2. 执行量化
        x_quantized = np.zeros_like(x)
        for i in range(len(x)):
            b = bits_array[i]
            if b <= 0:
                x_quantized[i] = 0.0 # 0比特，数据丢失
            else:
                levels = 2 ** b
                step = (2 * val_range) / levels
                x_quantized[i] = np.round(x[i] / step) * step
                x_quantized[i] = np.clip(x_quantized[i], -val_range, val_range)
                
        return x_quantized, bits_array

    def transmit(self, x_current, use_etc=True, threshold=0.1, 
                 use_quantization=False, weights=None, variances=None, B_total=16, val_range=3.14, custom_bits=None):
        """
        核心发送函数：串联 ETC 判断与数据压缩
        Returns:
            x_received (np.ndarray): 控制器最终收到的数据 (可能是零阶保持，也可能是粗糙的量化值)
            triggered (bool): 本次是否发生了真实的网络通信
            bits_used (int): 本次通信消耗的比特数
        """
        # 1. ETC 触发判定
        triggered = True
        if use_etc:
            triggered = self.check_event_trigger(x_current, threshold)

        if not triggered:
            # 不发包，节省带宽。接收端继续使用上次的旧数据 (零阶保持)
            return self.x_last_sent.copy(), False, 0

        # 2. 如果触发了发包，开始处理量化压缩
        x_to_send = x_current.copy()
        bits_used = 0

        if use_quantization:
            if custom_bits is not None:
                # 强化学习模型控制：使用外部动态传入的比特分配方案
                x_to_send, _ = self.lqr_weighted_quantize(
                    x_to_send, None, None, B_total, val_range, custom_bits=custom_bits)
                bits_used = int(np.sum(custom_bits))
            elif weights is not None and variances is not None:
                # LQR 静态加权压缩
                x_to_send, _ = self.lqr_weighted_quantize(
                    x_to_send, weights, variances, B_total, val_range)
                bits_used = B_total
            else:
                # 降级方案：传统均匀压缩
                uniform_weights = np.ones(self.num_dims)
                uniform_variances = np.ones(self.num_dims)
                x_to_send, _ = self.lqr_weighted_quantize(
                    x_to_send, uniform_weights, uniform_variances, B_total, val_range)
                bits_used = B_total
        else:
            # 不压缩，直接发 64-bit 浮点数 (理想情况)
            bits_used = self.num_dims * 64 

        # 3. 更新记忆与统计，完成发送
        self.x_last_sent = x_to_send.copy()
        self.total_transmissions += 1
        self.total_bits_used += bits_used

        return x_to_send, True, bits_used


# ==========================================
# 单元测试代码 (仅在此文件被直接运行时执行)
# ==========================================
if __name__ == "__main__":
    print("=== 开始单元测试: core/communication.py ===")
    
    # 假设机械臂有2个关节，仅传输位置 (num_dims=2)
    channel = CommunicationChannel(num_dims=2)
    
    # 模拟物理引擎产生的连续状态序列
    simulated_states = [
        np.array([0.0, 0.0]),   # Step 0
        np.array([0.05, 0.01]), # Step 1: 微微移动
        np.array([0.08, 0.02]), # Step 2: 微微移动
        np.array([0.30, 0.10]), # Step 3: 突然大幅移动 (需要触发)
        np.array([0.31, 0.11]), # Step 4: 再次微调
    ]
    
    # 设定敏感度矩阵P的对角线
    test_weights = np.array([1000.0, 1.0])
    test_variances = np.array([0.1, 0.1])
    test_budget = 8
    
    print(f"\n[测试环境设定] ETC阈值: 0.1, 总带宽: 8 bits, 敏感度权重: {test_weights}")
    print("-" * 60)
    
    for step, x_true in enumerate(simulated_states):
        # 通过信道传输 (使用LQR静态加权)
        x_recv, triggered, bits = channel.transmit(
            x_current=x_true,
            use_etc=True, threshold=0.1,
            use_quantization=True, weights=test_weights, variances=test_variances, B_total=test_budget, val_range=1.0
        )
        
        status = "📡 触发发送" if triggered else "🔇 保持沉默"
        print(f"Step {step}:")
        print(f"  真实数据: {x_true.round(3)}")
        print(f"  通信动作: {status} (耗费 {bits} bits)")
        print(f"  接收数据: {x_recv.round(3)}")
        print("-" * 60)
        
    print("\n=== 单元测试结束 ===")
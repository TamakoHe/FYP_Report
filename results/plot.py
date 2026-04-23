import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 加载与预处理数据
# ==========================================
raw_data_eval_err = pd.read_csv("/Users/hewenxiao/Documents/2026/FYP_Report/results/14_4/training_convergence_14bit.csv")
tot_data = 25

# 优化提取逻辑: 利用 pandas 的切片功能，告别 for 循环
# 直接将截取后的数据转换为 numpy 数组
eval_errs = raw_data_eval_err["Mean_Reward"].iloc[:tot_data].values
r_step = np.arange(tot_data)

# ==========================================
# 2. 设置全局精美样式
# ==========================================
# 使用自带的清爽网格主题
plt.style.use('seaborn-v0_8-whitegrid') 
plt.rcParams['figure.dpi'] = 300           # 设置高分辨率输出 (适合论文)
plt.rcParams['font.family'] = 'sans-serif' # 使用无衬线字体，现代感更强

# ==========================================
# 3. 创建画布与绘制曲线
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制主曲线，采用带有高科技感的蓝色，并增加线宽与带白色边缘的数据点
ax.plot(r_step, eval_errs, 
        color='#2563EB',          # 主线条颜色 (Tailwind Blue-600)
        linewidth=2.5,            # 增加线宽
        marker='o',               # 添加圆形标记
        markersize=8,             # 标记大小
        markerfacecolor='#FFFFFF',# 标记内部填充白色
        markeredgecolor='#2563EB',# 标记边缘设为线条同色
        markeredgewidth=2,
        label='RL Model Rewards')

# 添加曲线下方的半透明渐变填充效果，提升视觉层次与高级感
ax.fill_between(r_step, eval_errs, 0, color='#2563EB', alpha=0.1)

# ==========================================
# 4. 图表细节深度美化
# ==========================================
# 【修复1】：适当减小字号和 padding，防止标题过度外扩
ax.set_title('Rewards Over Steps', 
             fontsize=14, fontweight='bold', color='#1F2937', pad=12)
ax.set_xlabel('Steps', fontsize=12, fontweight='bold', color='#4B5563')
ax.set_ylabel('Rewards', fontsize=12, fontweight='bold', color='#4B5563')

# 调整刻度标签颜色与大小
ax.tick_params(axis='both', which='major', labelsize=11, colors='#6B7280')

# 优化背景网格线
ax.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)

# 移除顶部和右侧多余的边框线 (Spines)，让图表更加清爽通透
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#D1D5DB')
ax.spines['bottom'].set_color('#D1D5DB')

# 【修复2】：动态调整 Y 轴尺度，放弃从 0 开始，放大 600->400 的下降趋势
y_min = np.min(eval_errs)
y_max = np.max(eval_errs)
# 将顶部裕量调大到 20%，给图例腾出绝对安全的空间，绝不挡住曲线
y_margin = (y_max - y_min) * 0.20 
if y_margin == 0: y_margin = 10   # 防止常数数据报错
ax.set_ylim(bottom=y_min - (y_max - y_min)*0.1, top=y_max + y_margin)

# 美化图例 (修改位置为 best 自动避让曲线，并增加半透明背景)
ax.legend(loc='best', fontsize=11, frameon=True, fancybox=True, shadow=True, borderpad=0.8, framealpha=0.9)

# 【终极修复】：通过 rect 参数强制在画布顶部预留 4% 的绝对空间，彻底解决标题被裁切的问题
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
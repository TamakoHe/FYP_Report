import pybullet as p
import pybullet_data
import time
import math

def main():
    # 1. 启动 PyBullet GUI 可视化界面
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 设置重力和背景
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    
    # 调整摄像机视角，方便观察小型桌面机械臂
    p.resetDebugVisualizerCamera(cameraDistance=0.8, 
                                 cameraYaw=45, 
                                 cameraPitch=-30, 
                                 cameraTargetPosition=[0, 0, 0.2])

    print("\n" + "="*50)
    print(" 🤖 Jiobt1 5-DOF 定制机械臂 - 数字孪生可视化调试器")
    print("="*50)
    print("正在加载 URDF 模型...")

    # 2. 加载我们定制的 5轴 机械臂 URDF
    # 注意：确保 "custom_5dof_arm.urdf" 文件与本脚本在同一目录下
    try:
        robot_id = p.loadURDF("custom_5dof_arm.urdf", basePosition=[0, 0, 0], useFixedBase=True)
        print("✅ 模型加载成功！")
    except p.error:
        print("❌ 错误：找不到 custom_5dof_arm.urdf 文件！请确保它和 Python 脚本在同一目录下。")
        return

    # 3. 创建 UI 滑动条 (Sliders)，用于实时调试 5 个自由度
    # 针对纯旋转关节 (J0, J4) 放宽到 ±180度 (-3.14 到 3.14)，俯仰关节保留安全限制
    sliders = []
    joint_params = [
        {"name": "J0 (Base Yaw)", "min": -3.14, "max": 3.14},
        {"name": "J1 (Shoulder Pitch)", "min": -2.35, "max": 2.35},
        {"name": "J2 (Elbow Pitch)", "min": -2.35, "max": 2.35},
        {"name": "J3 (Wrist Pitch)", "min": -2.35, "max": 2.35},
        {"name": "J4 (Gripper Roll)", "min": -3.14, "max": 3.14}
    ]
    
    for param in joint_params:
        # 参数：名称，最小值，最大值，初始值
        slider = p.addUserDebugParameter(param["name"], param["min"], param["max"], 0.0)
        sliders.append(slider)

    # 在末端执行器上绘制一条辅助线，标示方向
    p.addUserDebugLine([0,0,0], [0,0,0.1], [1,0,0], 2, parentObjectUniqueId=robot_id, parentLinkIndex=6)

    print("\n🎮 调试器已启动！请拖动右侧控制面板的滑动条来控制机械臂的各个关节。")
    print("按 Ctrl+C 或关闭窗口退出程序。\n")

    # 4. 实时物理仿真循环
    try:
        while True:
            # 步进物理引擎
            p.stepSimulation()
            
            # 读取 5 个滑动条的当前数值
            target_angles = []
            for i in range(5):
                try:
                    # 加入 try-except 防止 Mac M系列芯片 Metal 渲染器初始帧未就绪导致的读取报错
                    angle = p.readUserDebugParameter(sliders[i])
                except p.error:
                    angle = 0.0 # 如果读取失败，暂时返回默认值 0.0
                target_angles.append(angle)
            
            # 将读取到的角度实时下发给机械臂的 5 个电机
            # 使用 POSITION_CONTROL 模式模拟真实的 PWM 位置控制舵机
            for i in range(5):
                p.setJointMotorControl2(bodyIndex=robot_id, 
                                        jointIndex=i, 
                                        controlMode=p.POSITION_CONTROL, 
                                        targetPosition=target_angles[i],
                                        force=10.0,      # 模拟舵机扭矩
                                        maxVelocity=3.0) # 模拟舵机最大速度
            
            # 保持界面刷新率 (60Hz 渲染以节省性能，底层物理 240Hz)
            time.sleep(1./60.)
            
    except KeyboardInterrupt:
        print("\n退出调试器。")
    finally:
        p.disconnect()

if __name__ == '__main__':
    main()
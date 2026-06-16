# EGO Teleop — 第一人称遥操作

无需 VR 头显，用 RealSense D435i 深度相机 + MediaPipe Hands 实现第一人称手部遥操作。

## 架构

```
📷 RealSense D435i (宿主机 conda)
    ↓ RGB 640×480 @ 30fps
🖐️ MediaPipe Hands — 21 关键点手部追踪
    ↓ 手腕归一化坐标 → 工作空间坐标
🌐 UDP JSON → Docker :9999
    ↓
🎯 mujoco_ego_sim.py (单文件，无 ROS2 依赖)
    ├── UDP 接收线程
    ├── MuJoCo 零空间正则化 IK（位置）
    ├── 位置伺服 + 临界阻尼 (kp=200, d=28)
    ├── MuJoCo 物理仿真 (8 steps/cycle, dt=0.002s)
    └── launch_passive 3D 可视化 (STL 网格)
```

## 核心设计

### 零空间正则化 IK

xArm6 有 6 个关节但位置目标只有 3 自由度，存在无穷多解。传统阻尼伪逆 IK 在不同帧会收敛到不同解，导致机械臂抖动。

解决方案：
- **零空间投影** `(I − J⁺J)` 在不影响末端位置的子空间内拉向 home 姿态
- **q_init = 当前仿真状态**，保证帧间连续性
- **阻尼伪逆** `J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹`，λ=0.1
- **最大步长限制** 0.08 rad/次，防止大跳变

### 位置伺服

使用 MuJoCo 的 `position` actuator，kp=200，临界阻尼 damping=28，每控制周期 8 次物理步进。

## 参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| kp | 200 | 伺服刚度 |
| damping | 28 | 关节阻尼（临界阻尼 2√200） |
| nullspace_gain | 0.03 | 零空间拉向 home 的力度 |
| IK damping | 0.1 | 数值 IK 阻尼 |
| IK max_step | 0.08 rad | 每次迭代最大步长 |
| physics_steps | 8/cycle | dt=0.002s × 8 = 0.016s |
| HOME | [0, -0.3, 0, 1.2, 0, 0] | 默认 home 姿态 |

## 手部→工作空间映射

| 手部坐标 (MediaPipe) | 机器人坐标 (xArm6) | 范围 |
|------|------|------|
| X (左右, 0–1) | Y (左右) | ±0.2m |
| Y (上下, 0–1) | Z (上下) | 0.15–0.45m |
| Z (深度, 0–1) | X (前后) | 0.05–0.35m |

## 启动

```bash
# 终端 1 — Docker 内启动仿真器
make exec cmd="bash /workspace/umi/stage_2/ego/scripts/restart_ego_sim.sh"

# 终端 2 — 宿主机启动手部追踪
conda activate ego
cd /home/aaron/workspace/umi
python stage_2/ego/mediapipe_ego.py --udp
```

控制键：`q` 退出

## 文件清单

```
stage_2/ego/
├── simulation/
│   ├── mujoco_ego_sim.py      # 主仿真器（单文件，自包含）
│   ├── mujoco_ik.py           # MuJoCo Jacobian IK 求解器
│   ├── xarm_mesh.xml          # STL 网格 MJCF 模型
│   └── *.STL                  # 7 个 STL 网格文件
├── scripts/
│   ├── restart_ego_sim.sh     # 仿真器重启脚本
│   ├── start_ego_pipeline.sh  # ROS2 管道启动（旧方案）
│   ├── start_ego_viewer.sh    # 独立 viewer 启动
│   ├── start_sim_viewer.sh    # 通用仿真 viewer
│   ├── start_xarm6_viewer.sh  # xArm6 viewer 启动
│   └── xarm6_viewer.py        # ROS2 订阅的独立 viewer
└── mediapipe_ego.py           # 宿主机端：RealSense + MediaPipe + UDP
```

## 模型

使用 `models/urdf2mjcf/xarm/` 下的 **STL 网格模型**转换为 MJCF 格式（`xarm_mesh.xml`），7 个 STL 文件精确还原 xArm6 外观。6 个关节 + 末端执行器 site。

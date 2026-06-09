# UMI — Universal Manipulation Interface

通用机器人操控接口：基于 Quest 3 VR 手部追踪的遥操作与模仿学习系统。

## 项目状态

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| Stage 1 | 月 1–3 | 硬件集成与遥操作回路 | ✅ 完成（35/36 测试通过，1 个时序敏感） |
| Stage 2 | 月 3–6 | 数据采集流水线 | ✅ 完成（MuJoCo 仿真、质量过滤、v3.0 转换、任务定义） |
| Stage 3 | 月 6–10 | 模型训练基线 | ✅ ACT + Diffusion Policy 已完成（5 个模型注册） |
| Stage 4 | 月 10+ | 部署与泛化 | 🔲 待开始 |

### 当前性能基线

| 指标 | 实测值 | 目标 |
|------|--------|------|
| 端到端延迟 | p50: 8.1ms | <50ms |
| IK 求解 | mean: 26ms, p99: 32ms | <33ms (30Hz) |
| 时间同步 | 687μs | <5ms |
| ACT 训练 Loss（state-only, 20K steps） | 0.60→0.15 | 收敛 |
| ACT 训练 Loss（state+visual, 10K steps） | 0.65→0.18 | 收敛 |
| DP 训练 Loss（state+visual, 5K steps） | 0.24→0.05 | 收敛 |
| FK 精度（MuJoCo vs 手写） | 0.000mm | — |

### 已注册模型（9 个）

| 模型 | 类型 | 参数 | 数据 | Loss | Test MSE |
|------|------|------|------|------|----------|
| act_diverse_20k | ACT | 9.8M | 300 eps 合成数据 | 0.154 | 0.90 |
| act_diverse_v2 | ACT | 9.8M | 400 eps 噪声合成 | 0.223 | 0.89 |
| act_teleop | ACT | 9.8M | 11 eps 人类遥操作 | 0.124 | 1.65 |
| act_mixed | ACT | 9.8M | 300 合成 + 11 人类 | 0.199 | 1.77 |
| dp_diverse_10k | Diffusion | 63.3M | 300 eps 合成数据 | 0.039 | 0.052 |
| act_visual | ACT | 9.8M | 300 eps（128-dim 视觉特征） | 0.184 | — |
| dp_visual | Diffusion | 63.3M | 300 eps（128-dim 视觉特征） | 0.053 | — |
| act_state_only | ACT | 9.8M | 300 eps（state-only 基线） | 0.184 | 0.851 |
| **act_goal** | **ACT** | **9.8M** | **400 eps（目标条件化）** | **0.167** | — |

### 目标条件化（Goal-Conditioned Policy）

最新进展：训练了目标条件化 ACT 策略，输入 `(current_state, goal_position) → action`，模型学会朝不同目标移动：

| 目标 | 起始距离 | 最小距离 | 改善 |
|------|----------|----------|------|
| Goal 0 (home) | 3.38 rad | 2.94 rad | 13% |
| Goal 3 (forward) | 4.01 rad | 2.63 rad | 34% |
| Goal 6 (far reach) | 3.76 rad | 1.59 rad | **58%** |

> **关键发现**：`goal_position`（6 维连续向量）条件化有效，而 `task_index`（one-hot）条件化完全无效——模型忽略了离散的任务索引。连续目标表示提供了可用的空间信息。

查看模型：`python3 stage_2/umi_pipeline.py models`
模型对比：`python3 stage_2/umi_pipeline.py compare`

## 快速开始（Docker）

```bash
# 1. 构建 Docker 镜像（首次 10-15 分钟）
make build

# 2. 编译 + 测试 + 启动 Mock 管道
make shell cmd="bash -c 'cd /ros2_ws && colcon build && \
  source install/setup.bash && \
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  python3 -m pytest /workspace/umi/stage_1/tests/ -v && \
  ros2 launch launch teleop_mock.launch.py'"
```

详细启动指南见 [`docs/START.md`](docs/START.md)。

## 仿真遥操作（键盘控制 + 3D 可视化）

无需硬件，在 MuJoCo 仿真中操控 xArm6 机械臂，支持实时录制。

```bash
# 启动容器
make up

# 终端 1：启动控制管道（后台）
make exec cmd="bash /workspace/umi/sim_start_pipeline.sh &"

# 终端 2：3D 可视化（STL 网格模型，彩色连杆）
make exec cmd="bash /workspace/umi/sim_viewer.sh"

# 终端 3：键盘操控（终端实时状态显示）
make exec cmd="bash /workspace/umi/sim_teleop.sh"
```

键盘控制：W/S 前后、A/D 左右、Q/E 升降、I/K 俯仰、J/L 横滚、U/O 偏航、空格夹爪、R 复位、Tab 录制切换

## UMI Pipeline（端到端训练流水线）

```bash
# 查看所有命令
python3 stage_2/umi_pipeline.py

# 快速验证（生成 10 eps + 训练 ACT + DP）
python3 stage_2/umi_pipeline.py validate

# 完整训练（生成 300 eps + 训练 ACT 20000 steps + DP 5000 steps）
python3 stage_2/umi_pipeline.py train --eps 300 --steps 20000

# 在已有数据上训练
python3 stage_2/umi_pipeline.py train --data data/my_dataset_v3 --steps 20000

# 对比所有模型
python3 stage_2/umi_pipeline.py compare

# 查看已注册模型
python3 stage_2/umi_pipeline.py models
```

### 数据生成

```bash
# 生成多样化训练数据（关节空间轨迹，MuJoCo 仿真）
python3 stage_2/generate_diverse_data.py -n 300 -o data/my_dataset --v3

# 轨迹类型：point_to_point、multi_waypoint、hold
# 特性：速度限制控制 + 高斯噪声 + 随机停顿 + 97% 关节范围覆盖
```

### 单独训练

```bash
# ACT 训练
python3 stage_2/train_act.py --data data/my_dataset_v3 --steps 20000 --batch-size 32

# Diffusion Policy 训练
python3 stage_2/train_dp.py --data data/my_dataset_v3 --steps 5000 --batch-size 32

# Rollout 评估（MuJoCo 中执行策略）
python3 stage_2/evaluate_rollout.py --checkpoint outputs/act_diverse/best.pt --runs 20
```

## 架构

```
键盘 / Quest3 手部追踪
    │ /hand/right/wrist_pose
    ▼
teleop_bridge (hand_mapper)     ← IK（手写 DH 参数 + 雅可比伪逆）
    │ /teleop/command/joints     ← IK 重复求解跳过 + 低通滤波
    ▼
safety (safety_guardian)        ← 关节限位/速度/急停 + 多后端（mock/xarm6/mujoco）
    │
    ▼
robot_hal (RobotInterface ABC)  ← Mock / xArm6 / MuJoCo
    │
    ▼
recorder (HDF5 录制)            ← 同步多模态数据
    │
    ▼
LeRobot v3.0 (Parquet + JSON)   ← 数据导出
    │
    ▼
ACT / Diffusion Policy          ← 端到端策略训练
    │
    ▼
Model Registry                  ← 模型元数据 + 对比评估
```

### 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| `kinematics` | `stage_1/kinematics/` | FK/IK/Jacobian（纯 Python，不依赖 ROS） |
| `robot_hal` | `stage_1/robot_hal/` | 硬件抽象层（Mock/XArm6/MuJoCo） |
| `teleop_bridge` | `stage_1/teleop_bridge/` | Quest3 → ROS2 桥接 + 手眼标定 + IK 震荡抑制 |
| `perception` | `stage_1/perception/` | USB 相机驱动 |
| `safety` | `stage_1/safety/` | 安全守护节点（mock/xarm6/mujoco 后端） |
| `recorder` | `stage_1/recorder/` | HDF5 录制 + LeRobot v2.x/v3.0 转换 |
| `launch` | `stage_1/launch/` | 全系统启动文件 |
| `stage_2` | `stage_2/` | 数据管道 + MuJoCo 仿真 + ACT/DP 训练 + 评估 |

### Stage 2 工具清单

| 工具 | 文件 | 功能 |
|------|------|------|
| 数据生成 | `generate_diverse_data.py` | MuJoCo 关节空间轨迹生成（3 种类型，速度限制，噪声） |
| ACT 训练 | `train_act.py` | ACT 训练脚本（GPU/CPU, cosine LR, checkpoint） |
| DP 训练 | `train_dp.py` | Diffusion Policy 训练（UNet 63M, denoising loss） |
| Rollout 评估 | `evaluate_rollout.py` | MuJoCo 中执行策略，计算成功率 |
| 模型对比 | `compare_models.py` | 多模型 test MSE 对比 |
| 流水线入口 | `umi_pipeline.py` | 统一 CLI（validate/train/compare/models） |
| 流水线验证 | `validate_pipeline.py` | 端到端小规模完整性检查 |
| v3.0 转换 | `lerobot_v3_converter.py` | HDF5 → LeRobot v3.0 Parquet 格式 |
| 质量过滤 | `quality_filters.py` | 9 项数据质量检查 |
| 任务管理 | `task_manager.py` | 任务定义 + tasks.parquet 导出 |
| 时间同步 | `time_sync_check.py` | 时间同步精度评估 |
| ACT 研究 | `act_research.py` | mock 批量录制 + v3 转换 + 格式验证 |
| 键盘遥操作 | `simulation/keyboard_teleop.py` | 键盘控制 MuJoCo 机械臂 |
| 3D 可视化 | `simulation/viewer_node.py` | MuJoCo 窗口渲染 |
| MJCF 模型 | `simulation/xarm6.xml` | 精确 DH 模型（14 bodies, FK 误差 0mm） |
| 多任务数据 | `generate_multitask_data.py` | 任务条件化数据生成（5 个到达任务） |
| 目标条件数据 | `generate_goal_data.py` | 目标条件化数据生成（8 个目标，400 eps） |
| 多任务训练 | `train_multitask_act.py` | 任务条件化 ACT 训练 |
| 目标条件训练 | `train_goal_act.py` | 目标条件化 ACT 训练（state+goal → action） |
| 多任务评估 | `evaluate_multitask.py` | 多任务/目标条件化策略 rollout 评估 |

详细架构见 [`CLAUDE.md`](CLAUDE.md)，阶段一模块文档见 [`stage_1/README.md`](stage_1/README.md)。

## 测试

```bash
# 纯 Python 测试（宿主机，无需 Docker）：
PYTHONPATH=. python3 -m pytest stage_1/tests/test_fk.py stage_1/tests/test_ik.py \
       stage_1/tests/test_jacobian.py stage_1/tests/test_mock_robot.py \
       stage_1/tests/test_hdf5_writer.py stage_1/tests/test_lerobot_converter.py -v

# 完整 ROS2 集成测试（容器内）：
make shell cmd="bash -c 'cd /ros2_ws && colcon build && \
  source install/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  python3 -m pytest /workspace/umi/stage_1/tests/ -v \
  -p no:launch_testing -p no:launch_testing_ros'"
```

## 关键修复记录

- **IK 振荡**（#b1d70b1）：缓存上一帧目标位姿，pose_error < 5mm 时跳过 IK 求解，消除关节抖动
- **键盘操控方向**（#2cd7f02, #32d788d）：使用 identity 标定，直接以机器人坐标系控制
- **仿真机械臂 FK 误差 0mm**（MJCF v2）：两层 body 结构精确建模 DH 运动学

## 规划

项目整体规划见 [`planning/full_planning/full_v1.md`](planning/full_planning/full_v1.md)。

每日任务规划与完成记录见 `planning/tomorrow_*.md`。

## 技术栈

- **ROS2 Humble** (Ubuntu 22.04)
- **Python 3.10**
- **MuJoCo** 物理仿真
- **LeRobot** (HuggingFace) 数据格式 + 策略 API
- **ACT** (Action Chunking Transformer) 9.8M 参数
- **Diffusion Policy** (UNet) 63.3M 参数
- **Quest 3** VR 手部追踪
- **xArm6** 机械臂

## 许可证

MIT

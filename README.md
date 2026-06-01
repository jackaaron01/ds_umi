# UMI — Universal Manipulation Interface

通用机器人操控接口：基于 Quest 3 VR 手部追踪的遥操作与模仿学习系统。

## 项目状态

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| Stage 1 | 月 1–3 | 硬件集成与遥操作回路 | ✅ 完成（36/36 测试通过） |
| Stage 2 | 月 3–6 | 数据采集流水线 | 🔄 进行中（仿真环境、质量过滤、格式转换已就绪） |
| Stage 3 | 月 6–10 | ACT 模型训练 | 🔲 待开始 |
| Stage 4 | 月 10+ | 部署与泛化 | 🔲 待开始 |

### 当前性能基线

| 指标 | 实测值 | 目标 |
|------|--------|------|
| 端到端延迟 | p50: 8.1ms | <50ms |
| IK 求解 | 26ms | <33ms (30Hz) |
| 时间同步 | 687μs | <5ms |

## 快速开始

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

## 架构

```
Quest3 手部追踪
    │ UDP
    ▼
teleop_bridge (hand_mapper)     ← IK (手写 DH 参数 + 雅可比伪逆)
    │ /teleop/command/joints
    ▼
safety (safety_guardian)        ← 关节限位/速度/急停
    │
    ▼
robot_hal (RobotInterface ABC)  ← Mock / xArm6 / MuJoCo
    │
    ▼
recorder (HDF5 录制)            ← 同步多模态数据
    │
    ▼
LeRobot v3.0 (Parquet + JSON)   ← 数据导出
```

### 模块

| 模块 | 路径 | 说明 |
|------|------|------|
| `kinematics` | `stage_1/kinematics/` | FK/IK/Jacobian（纯 Python，不依赖 ROS） |
| `robot_hal` | `stage_1/robot_hal/` | 硬件抽象层（Mock/XArm6/MuJoCo） |
| `teleop_bridge` | `stage_1/teleop_bridge/` | Quest3 → ROS2 桥接 + 手眼标定 |
| `perception` | `stage_1/perception/` | USB 相机驱动 |
| `safety` | `stage_1/safety/` | 安全守护节点 |
| `recorder` | `stage_1/recorder/` | HDF5 录制 + LeRobot 转换 |
| `launch` | `stage_1/launch/` | 全系统启动文件 |
| `stage_2` | `stage_2/` | 数据管道工具 + MuJoCo 仿真 |

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

## 规划

项目整体规划见 [`planning/full_planning/full_v1.md`](planning/full_planning/full_v1.md)。

每日任务规划与完成记录见 `planning/tomorrow_*.md`。

## 技术栈

- **ROS2 Humble** (Ubuntu 22.04)
- **Python 3.10**
- **MuJoCo** 仿真
- **LeRobot** (HuggingFace) 数据格式
- **Quest 3** VR 手部追踪
- **xArm6** 机械臂

## 许可证

MIT

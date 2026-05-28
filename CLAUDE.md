# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

UMI（Universal Manipulation Interface，通用操控接口）是一个用于机器人操控的遥操作与模仿学习系统。它使用 Quest 3 VR 头显进行手部追踪来控制机械臂（xArm6），录制多模态演示数据，并最终训练端到端策略（ACT、Diffusion Policy、VLA 微调）。

项目当前处于**第一阶段**：硬件集成与遥操作回路 —— 实现实时控制和同步录制。

## 开发环境

所有开发都在 Docker 中进行。镜像基于 `osrf/ros:humble-desktop-full`（Ubuntu 22.04，ROS2 Humble LTS）。Dockerfile 位于 `docker/Dockerfile`，compose 编排文件位于 `docker/docker-compose.yml`。

```bash
make build       # 构建 Docker 镜像（--no-cache，完整重新构建）
make rebuild     # 利用 Docker 层缓存重新构建（增量修改时更快）
make shell       # 进入容器的交互式 shell
make shell cmd="pytest"  # 在新容器中执行命令，执行完毕后退出
make up          # 后台启动容器（detached 模式）
make down        # 停止并删除容器
make exec cmd="ros2 topic list"  # 在已运行的容器中执行命令
make logs        # 跟踪容器日志
make clean       # 删除构建缓存和悬空镜像
```

容器使用 **host 网络模式**（ROS2 DDS 发现和 Quest3 UDP 数据流所必需），将项目根目录挂载到 `/workspace/umi`，并转发 X11 以支持 GUI 应用（RViz、PlotJuggler）。

**第一次使用流程**：
```bash
make build                              # 1. 构建镜像（只需一次）
make shell                              # 2. 进入容器
# （容器内）
sudo pip install --upgrade "setuptools>=65,<80"  # 3. 降级 setuptools（兼容 colcon-core）
colcon build --symlink-install          # 4. 编译 ROS2 包（源码修改后也需要重新执行）
source /ros2_ws/install/setup.bash      # 5. source 编译产物（entrypoint 已写入 .bashrc，新 shell 自动执行）
# CycloneDDS 未默认安装，若 ros2 命令报错 rmw_cyclonedds_cpp not found：
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

运行测试：`make shell cmd="pytest"`（在容器内执行）。

**重要：`make shell cmd="..."` 每次调用创建新容器**（`--rm`），之前安装的依赖和编译产物不会保留。需要将多个操作（编译 + 测试）合并到一个命令中。

```bash
# 完整的构建 + 测试流程（容器内一步完成）：
make shell cmd="bash -c 'sudo pip install -q --upgrade \"setuptools>=65,<80\" && cd /ros2_ws && colcon build --symlink-install && source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && pytest /workspace/umi/stage_1/tests/ -v'"
```

**setuptools 版本问题**：默认容器内 setuptools 80.x 与 colcon-core 不兼容，需先 `sudo pip install --upgrade "setuptools>=65,<80"`。

**ROS2 DDS**：docker-compose.yml 配置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，但该包未默认安装。容器内实际可用的是 FastRTPS（`rmw_fastrtps_cpp`）。如需 CycloneDDS，安装 `ros-humble-rmw-cyclonedds-cpp`。`ROS_DOMAIN_ID` 默认为 0。

## 架构

### 两个代码层

**根目录**（`robot_hal/`, `kinematics/`）—— 纯 Python 模块（仅有 `__init__.py`，定义抽象接口和数据类），不依赖 ROS2，被 `PYTHONPATH` 直接引用。不要在这些根目录中新增实现代码。

**根目录 ROS2 包壳体**（`teleop_bridge/`, `perception/`, `safety/`, `recorder/`, `launch/`）—— 每个目录仅包含一个嵌套的同名 Python 包（含 docstring `__init__.py`）。`entrypoint.sh` 将这些目录符号链接到 `/ros2_ws/src/` 供 colcon 发现。`setup.py`、`package.xml` 和所有实现代码位于 `stage_1/<pkg>/` 中。

**`stage_1/` 目录** —— 所有第一阶段的实现代码都在这里。`stage_1/kinematics/`、`stage_1/robot_hal/` 等包含实际的 Python 模块。ROS2 包通过其 `setup.py` 中的 `package_dir` 映射机制引用 stage_1 代码。

### 两类包

**ROS2 包**（由 colcon 构建，通过 `entrypoint.sh` 以符号链接方式链接到 `/ros2_ws/src/`）：
- `teleop_bridge/` — Quest3 手部追踪 → ROS2 topics（关节/夹爪指令）。实现：`stage_1/teleop_bridge/`
- `perception/` — USB 相机 ROS2 驱动（OpenCV `VideoCapture`，可配置设备 ID/分辨率/FPS/标定文件），发布 `sensor_msgs/Image` + `CameraInfo`。实现：`stage_1/perception/camera_node.py`
- `safety/` — 安全守护节点：关节限位、速度/力矩异常检测、紧急停止、虚拟围栏。实现：`stage_1/safety/`
- `recorder/` — 同步 HDF5 录制（关节状态 + 夹爪 + 遥操作指令 + 可选的相机图像）；支持 `convert_to_lerobot` 命令导出为 LeRobot 标准格式（HDF5 + Parquet 元数据）。实现：`stage_1/recorder/`
- `launch/` — 全系统的 `ros2 launch` 文件（Mock 模式、真实硬件模式、仅录制模式）。实现：`stage_1/launch/`

**纯 Python 模块**（不依赖 ROS2，可在任何环境测试）：
- `robot_hal/` — 硬件抽象层：`RobotInterface` 抽象基类、`JointState` 和 `GripperState` 数据类均定义在 `robot_hal/__init__.py` 中。具体实现在 `stage_1/robot_hal/`（`MockRobotInterface`、`XArm6Interface`）。所有上层代码仅依赖该抽象接口。
- `kinematics/` — 手写正运动学/逆运动学（DH 参数、雅可比伪逆），实现位于 `stage_1/kinematics/`，以 Pinocchio 作为参考答案进行交叉验证。

### setup.py 中的 package_dir 映射

每个 ROS2 包的 `setup.py`、`package.xml` 和 `resource/` 文件都位于 `stage_1/<pkg>/` 目录中。`package_dir` 使用 `"."` 将 `stage_1_<pkg>` Python 包映射到当前目录（即 `stage_1/<pkg>/`），因为 setup.py 与实现代码在同一目录下。

```python
# stage_1/teleop_bridge/setup.py 示例
setup(
    name="teleop_bridge",
    version="0.1.0",
    packages=["stage_1_teleop_bridge"],
    package_dir={"stage_1_teleop_bridge": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/teleop_bridge"]),
        ("share/teleop_bridge", ["package.xml"]),
    ],
    entry_points={
        "console_scripts": [
            "mock_hand_tracker = stage_1_teleop_bridge.mock_hand_tracker:main",
            "hand_mapper = stage_1_teleop_bridge.hand_mapper:main",
            "hand_tracking_node = stage_1_teleop_bridge.hand_tracking_node:main",
            "hand_eye_calibrate = stage_1_teleop_bridge.calibrate:main",
        ],
    },
)
```

这种模式应用于每一个含有 stage_1 实现的 ROS2 包。启动脚本条目指向 `stage_1_<pkg>` 命名空间。

### 数据流（遥操作管道）

```
Quest3 (Hand Tracking Streamer App) 或 mock_hand_tracker 节点
  → UDP/WiFi（Quest3）或合成数据（mock）
  → ROS2 topics: /hand/{hand}/keypoints, /hand/{hand}/wrist_pose
teleop_bridge/hand_mapper 节点
  → IK → low-pass filter → ROS2 topics: /teleop/command/joints, /teleop/command/gripper
safety/safety_guardian 节点（验证指令是否在安全范围内）
  → 转发给 robot_hal/RobotInterface 实现（MockRobotInterface 或 XArm6Interface）
recorder/recorder 节点（订阅 command + state + 可选的 camera topics，写入 HDF5）
perception/camera_node（可选，USB 相机 → sensor_msgs/Image）
```

### ROS2 Topic 合约

hand_mapper 通过 `hand` 参数支持 `left`/`right`，topic 前缀会自动对应 `/hand/{hand}/...`。

| Topic | 类型 | 发布者 | 订阅者 |
|-------|------|--------|--------|
| `/hand/{hand}/wrist_pose` | `PoseStamped` | mock_tracker / hand-tracking-sdk | hand_mapper |
| `/hand/{hand}/keypoints` | `Float32MultiArray` | mock_tracker / hand-tracking-sdk | hand_mapper |
| `/teleop/command/joints` | `JointState` | hand_mapper | safety, recorder |
| `/teleop/command/gripper` | `Float64` | hand_mapper | safety, recorder |
| `/teleop/state/joints` | `JointState` | safety | recorder |
| `/teleop/state/gripper` | `Float64` | safety | recorder |
| `/safety/status` | `String` | safety | (monitoring) |
| `/camera/rgb` | `Image` | camera_node | recorder (可选) |
| `/camera/depth` | `Image` | camera_node | recorder (可选) |

### HDF5 录制格式

Recorder 使用**层级 key 结构**（必须包含 `/`），HDF5Writer 跳过不含 `/` 的 key：

| HDF5 Key | 数据类型 | 描述 |
|----------|---------|------|
| `joint_command/position` | float64[6] | 遥操作指令关节角 |
| `joint_state/position` | float64[6] | 实际关节角 |
| `joint_state/velocity` | float64[6] | 实际关节速度 |
| `gripper/command` | float64 | 夹爪指令 (0-1) |
| `gripper/state` | float64 | 实际夹爪开度 |
| `sensors/camera/rgb` | uint8[H,W,3] | RGB 图像（可选） |
| `sensors/camera/depth` | uint8[H,W] | 深度图（可选） |

### LeRobot 格式导出

```bash
# 将录制的 HDF5 文件转换为 LeRobot 标准格式
convert_to_lerobot --input /tmp/umi_recordings --output /tmp/lerobot_export --features
```

Key 映射关系：`joint_command/position` → `action/joint_position`，`joint_state/position` → `observation/joint_position`，`gripper/command` → `action/gripper`，`sensors/camera/rgb` → `observation/images/camera_rgb` 等。输出包含 `meta/episodes.parquet`（episode_index + length）和可选的 `meta/features.json`。

### 实现阶段与构建顺序

| 阶段 | 内容 | 依赖 |
|------|------|------|
| 1 | `stage_1/kinematics/` — FK、IK、Jacobian（纯 Python，无 ROS） | 无 |
| 2 | `stage_1/robot_hal/` — MockRobotInterface、XArm6Interface | 阶段 1，根目录 robot_hal ABC |
| 3 | `stage_1/teleop_bridge/` — hand_mapper 节点、mock_hand_tracker 节点 | 阶段 1 |
| 4 | `stage_1/recorder/` — HDF5 writer + recorder ROS2 节点 | ROS2 |
| 5 | `stage_1/safety/` — safety guardian 节点 | 阶段 2 |
| 6 | `stage_1/launch/` — 3 个 launch 文件（mock、real、record-only） | 阶段 3–5 |
| 7 | `stage_1/tests/` — 单元测试 + 集成测试 | 阶段 1–5 |

阶段 1 和阶段 2 可以在宿主机上纯 Python 测试，无需 Docker。阶段 3–7 需要 ROS2，必须在容器内运行。

### 关键设计决策

- **HAL 是最重要的模块。** `robot_hal/RobotInterface` 是所有上层代码依赖的唯一抽象。从 xArm6 换到 UR5 或仿真器，只需写一个新的 `RobotInterface` 实现 —— 其他代码零改动。
- **kinematics/ 和 robot_hal/ 刻意不依赖 ROS。** 这样可以在 CI/notebook/非 ROS 环境中测试，也避免了框架锁定。
- **HDF5 是内部录制格式**（支持分层存储、压缩、随机访问、与 LeRobot 兼容）。
- **Pinocchio** 作为参考答案安装，用于验证手写运动学的正确性。
- **LeRobot**（HuggingFace）提供数据集格式和训练框架集成点，供后续阶段使用。
- **`launch` 包名冲突**：我们的 `launch/` ROS2 包与 ROS2 核心的 `launch` Python 模块同名。当 workspace 的 PYTHONPATH 覆盖系统路径时，ROS2 的 `launch_testing` pytest 插件会因找不到 `launch.actions` 而崩溃。运行 pytest 时需禁用该插件（`-p no:launch_testing`），或在 workspace setup.bash 之后 source 系统 ROS2 setup（让系统 `launch` 优先）。

### Docker 构建分层（顺序影响缓存）

Dockerfile（`docker/Dockerfile`）各层按修改频率排序：基础镜像 → 系统 apt 包 → ROS2 apt 包 → Pinocchio → Python 包（numpy/scipy/h5py/matplotlib/pytest → PyTorch → lerobot/hand-tracking-sdk/xarm-python-sdk → ipython/black/ruff）→ 创建用户 → 工作空间初始化 → 入口脚本。避免修改靠前的层。

## 项目规划

完整规划见 `planning/full_planning/full_v1.md`。当前阶段的详细规划见 `planning/stage1_planning/stage1_v1.md`。

- **第一阶段**（第 1–3 个月）：硬件集成 —— 遥操作回路、实时控制、同步录制
- **第二阶段**（第 3–6 个月）：数据管道 —— 时间同步（<5ms）、质量控制、任务设计
- **第三阶段**（第 6–10 个月）：模型训练 —— ACT 基线、Diffusion Policy、VLA 微调
- **第四阶段**（第 10 个月起）：部署 —— 在线微调、多任务泛化

## 测试

```bash
# 纯 Python 模块（kinematics, robot_hal, hdf5_writer, lerobot_converter）—— 可在宿主机直接运行：
pytest stage_1/tests/test_fk.py stage_1/tests/test_ik.py stage_1/tests/test_jacobian.py \
       stage_1/tests/test_mock_robot.py stage_1/tests/test_hdf5_writer.py \
       stage_1/tests/test_lerobot_converter.py -v

# 完整测试套件（包括 ROS2 集成测试）—— 必须在容器内运行：
# 注意：以下命令合并了 setuptools 降级、colcon 编译、workspace source 三步，
# 因为 make shell cmd="..." 每次创建新容器（--rm），之前的状态不保留。
make shell cmd="bash -c 'sudo pip install -q --upgrade \"setuptools>=65,<80\" && cd /ros2_ws && colcon build --symlink-install && source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && pytest /workspace/umi/stage_1/tests/ -v'"

# 仅运行纯 Python 测试（在容器内）：
make shell cmd="bash -c 'pytest /workspace/umi/stage_1/tests/test_fk.py /workspace/umi/stage_1/tests/test_ik.py /workspace/umi/stage_1/tests/test_jacobian.py /workspace/umi/stage_1/tests/test_mock_robot.py /workspace/umi/stage_1/tests/test_hdf5_writer.py /workspace/umi/stage_1/tests/test_lerobot_converter.py -v'"

# 通过 colcon 运行（ROS2 方式）：
make shell cmd="bash -c 'sudo pip install -q --upgrade \"setuptools>=65,<80\" && colcon test --packages-select teleop_bridge recorder safety'"

# 验证 FK 与 Pinocchio 的一致性（在容器内，Pinocchio 已安装）：
make shell cmd="python3 -c '
import pinocchio
import numpy as np
from stage_1.kinematics import forward_kinematics
q = np.random.uniform(-1, 1, 6)
positions, _ = forward_kinematics(q)
# 与 pinocchio 比较 ...
'"
```

### 手动端到端测试（Mock 模式）

```bash
# 终端 1：启动全系统
make shell
# （容器内，首次运行需要以下命令）
sudo pip install -q --upgrade "setuptools>=65,<80"
cd /ros2_ws && colcon build --symlink-install
source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 launch launch teleop_mock.launch.py

# 终端 2：触发录制
make exec cmd="ros2 service call /recorder/start std_srvs/srv/Trigger"

# 终端 3：检查 topics
make exec cmd="ros2 topic list"
make exec cmd="ros2 topic echo /teleop/state/joints --once"
```

## 代码检查与格式化

```bash
# 在容器内执行：
black .          # 格式化 Python 代码
ruff check .     # 静态检查 Python 代码
```

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

UMI（Universal Manipulation Interface，通用操控接口）是一个用于机器人操控的遥操作与模仿学习系统。它使用 Quest 3 VR 头显进行手部追踪来控制机械臂（xArm6），录制多模态演示数据，并最终训练端到端策略（ACT、Diffusion Policy、VLA 微调）。

项目当前处于**第二阶段完成、第三阶段起步**：遥操作回路已闭环（36/36 测试通过），仿真环境已集成（MuJoCo + 键盘遥操作 + 3D 可视化），数据管道工具（质量过滤、LeRobot v3.0 转换、任务定义、时间同步）已就绪，ACT 训练回路已验证（GPU/CPU 均可，loss 正常收敛）。

详细模块文档见 `stage_1/README.md`，启动指南见 `docs/START.md`。

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
cd /ros2_ws && colcon build             # 4. 编译 ROS2 包（源码修改后也需要重新执行）
source /ros2_ws/install/setup.bash      # 5. source 编译产物（entrypoint 已写入 .bashrc，新 shell 自动执行）
# CycloneDDS 未默认安装，若 ros2 命令报错 rmw_cyclonedds_cpp not found：
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

**不要使用 `--symlink-install`**：`package_dir={"stage_1_<pkg>": "."}` 中的 `"."` 会导致 colcon 在 Docker overlayfs 上报 `OSError: Invalid argument`。直接用 `colcon build`（复制模式），源码修改后重新 build 即可。

运行测试：需在容器内完成构建+测试（见下方完整命令）。`make shell cmd="..."` 每次创建新容器，依赖和编译产物不保留。

**重要：`make shell cmd="..."` 每次调用创建新容器**（`--rm`），之前安装的依赖和编译产物不会保留。需要将多个操作（编译 + 测试）合并到一个命令中。

```bash
# 完整的构建 + 测试流程（容器内一步完成）：
make shell cmd="bash -c 'sudo pip install -q --upgrade \"setuptools>=65,<80\" && sudo pip install -q \"anyio<4\" && cd /ros2_ws && colcon build && source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && sudo bash -c \"echo \\\"# no-op\\\" > /opt/ros/humble/lib/python3.10/site-packages/launch_testing_ros_pytest_entrypoint.py\" && python3 -m pytest /workspace/umi/stage_1/tests/ -v -p no:launch_testing -p no:launch_testing_ros'"
```

**pytest 插件冲突**：`launch_testing_ros` 在 ROS2 Humble 中注册了现代 pytest 不存在的 hook。使用 `-p no:launch_testing -p no:launch_testing_ros` 禁用。若仍有问题，用 `echo "# no-op" | sudo tee /opt/ros/humble/lib/python3.10/site-packages/launch_testing_ros_pytest_entrypoint.py` 与 `sudo pip install "anyio<4"`。

**setuptools 版本问题**：默认容器内 setuptools 80.x 与 colcon-core 不兼容，需先 `sudo pip install --upgrade "setuptools>=65,<80"`。

**ROS2 DDS**：docker-compose.yml 配置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，但该包未默认安装。容器内实际可用的是 FastRTPS（`rmw_fastrtps_cpp`）。如需 CycloneDDS，安装 `ros-humble-rmw-cyclonedds-cpp`。`ROS_DOMAIN_ID` 默认为 0。

## 架构

### 两个代码层

**根目录**（`robot_hal/`, `kinematics/`）—— 纯 Python 模块（仅有 `__init__.py`，定义抽象接口和数据类），不依赖 ROS2，被 `PYTHONPATH` 直接引用。不要在这些根目录中新增实现代码。

**根目录 ROS2 包壳体**（`teleop_bridge/`, `perception/`, `safety/`, `recorder/`, `launch/`）—— 每个目录仅包含一个嵌套的同名 Python 包（含 docstring `__init__.py`）。`entrypoint.sh` 将 `stage_1/<pkg>/` 符号链接到 `/ros2_ws/src/` 供 colcon 发现。`setup.py`、`package.xml` 和所有实现代码位于 `stage_1/<pkg>/` 中。不要在根目录的壳体中新增实现代码。

**`stage_1/` 目录** —— 所有第一阶段的实现代码都在这里。`stage_1/kinematics/`、`stage_1/robot_hal/` 等包含实际的 Python 模块。ROS2 包通过其 `setup.py` 中的 `package_dir` 映射机制引用 stage_1 代码。

### 两类包

**ROS2 包**（由 colcon 构建，通过 `entrypoint.sh` 以符号链接方式链接到 `/ros2_ws/src/`）：
- `teleop_bridge/` — Quest3 手部追踪 → ROS2 topics（关节/夹爪指令）。实现：`stage_1/teleop_bridge/`
- `perception/` — USB 相机 ROS2 驱动（OpenCV `VideoCapture`，可配置设备 ID/分辨率/FPS/标定文件），发布 `sensor_msgs/Image` + `CameraInfo`。实现：`stage_1/perception/camera_node.py`
- `safety/` — 安全守护节点：关节限位、速度/力矩异常检测、紧急停止、虚拟围栏。支持 `robot_mode` 参数：`mock`、`xarm6`、`mujoco`（MuJoCo 仿真）。实现：`stage_1/safety/`
- `recorder/` — 同步 HDF5 录制（关节状态 + 夹爪 + 遥操作指令 + 可选的相机图像）；支持 `convert_to_lerobot` 命令导出为 LeRobot v2.x（HDF5）或 v3.0（Parquet + JSON）格式。实现：`stage_1/recorder/`。v3.0 转换逻辑在 `stage_2/lerobot_v3_converter.py`
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

Recorder 使用**层级 key 结构**（必须包含 `/`），HDF5Writer 跳过不含 `/` 的 key。每 episode 数据包装在 `/episode_XXXXXX/` 组下，多 episode 可共存于单个 HDF5 文件。每个 data key 有对应的 `_timestamp` 数据集记录写入时间。

| HDF5 Key（相对于 episode 组） | 数据类型 | 描述 |
|----------|---------|------|
| `joint_command/position` | float64[N,6] | 遥操作指令关节角 |
| `joint_state/position` | float64[N,6] | 实际关节角 |
| `joint_state/velocity` | float64[N,6] | 实际关节速度 |
| `gripper/command` | float64[N,1] | 夹爪指令 (0-1) |
| `gripper/state` | float64[N,1] | 实际夹爪开度 |
| `sensors/camera/rgb` | uint8[N,H,W,3] | RGB 图像（可选） |
| `sensors/camera/depth` | uint8[N,H,W] | 深度图（可选） |

### LeRobot 格式导出

支持两种输出格式：

```bash
# v2.x 兼容格式（HDF5，默认）：
convert_to_lerobot --input /tmp/umi_recordings --output /tmp/lerobot_export --features

# v3.0 格式（Parquet + JSON）：
convert_to_lerobot --input /tmp/umi_recordings --output /tmp/lerobot_export --v3 --tasks stage_2/tasks/example_tasks.yaml
```

**v3.0 输出结构**：`data/chunk-000/file-000.parquet`（帧数据，含 `episode_index`/`frame_index`/`index`/`timestamp`/`task_index` 及 action/observation 列）、`meta/info.json`（数据集配置）、`meta/stats.json`（归一化统计量）、`meta/tasks.parquet`（任务定义）、`meta/episodes/`（episode 元数据）。Feature 命名使用点分隔（`action.joint_position`、`observation.joint_position`），而非斜杠。

**注意**：LeRobot v3.0 已将标准格式从 HDF5 迁移到 Parquet + MP4。默认的 v2.x HDF5 输出仅用于向后兼容。

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
- **IK seed 传播**：`hand_mapper` 在 IK 失败时仍将 best-effort 解作为下一帧的种子（`q_current`），避免级联失败。初始种子设为 `[0, -0.5, 0, 1.5, 0, 0]`（肘部弯曲的典型姿态），不采用 `zeros(6)`。`max_iterations` 从默认 200 降至 80，配合运行种子可在 ~26ms 内收敛。
- **pose_error 坐标系**：`pose_error()` 计算世界坐标系下的旋转误差（`R_err = R_des @ R_cur^T`），与雅可比矩阵的角速度列坐标系一致。这是 IK 收敛性的关键修正。
- **MuJoCo 仿真**：`MujocoRobotInterface`（`stage_2/simulation/mujoco_interface.py`）实现 `RobotInterface` ABC，将 MuJoCo 物理仿真集成到 safety_guardian 控制回路中。safety_node 设置 `robot_mode="mujoco"` 即可切换到仿真模式。MJCF 模型文件位于 `stage_2/simulation/xarm6.xml`（14 bodies），由 `generate_mjcf.py` 生成。MuJoCo 已添加到 Docker 镜像中（`pip install mujoco`），无需每次手动安装。
  - **MJCF 采用两层 body 结构**（rotor + link per joint）来精确建模 DH 运动学——单层 body 无法处理 a ≠ 0 关节的平移对关节角的依赖。`generate_mjcf.py` 使用纯 numpy（不依赖 scipy）计算旋转矩阵和 Euler 角。
  - **FK 精度**：MuJoCo FK 与手写 FK 在随机配置下误差 **0.000mm**（之前简化模型误差 >1m）。验证：`cd stage_2/simulation && python3 generate_mjcf.py`。

### Docker 构建分层（顺序影响缓存）

Dockerfile（`docker/Dockerfile`）各层按修改频率排序：基础镜像 → 系统 apt 包 → ROS2 apt 包 → Pinocchio → Python 包（numpy/scipy/h5py/matplotlib/pytest → PyTorch → lerobot/hand-tracking-sdk/xarm-python-sdk → ipython/black/ruff）→ 创建用户 → 工作空间初始化 → 入口脚本。避免修改靠前的层。

## 项目规划

GitHub 仓库：`https://github.com/jackaaron01/ds_umi`（本地 `master` → 远程 `main`）。

完整规划见 `planning/full_planning/full_v1.md`。当前阶段的详细规划见 `planning/stage1_planning/stage1_v1.md`。

- **第一阶段**（第 1–3 个月）：硬件集成 —— 遥操作回路、实时控制、同步录制（**已完成**）
- **第二阶段**（第 3–6 个月）：数据管道 —— 时间同步（<5ms）、质量控制、任务设计（**准备中**，部分工具已就绪）
- **第三阶段**（第 6–10 个月）：模型训练 —— ACT 基线、Diffusion Policy、VLA 微调
- **第四阶段**（第 10 个月起）：部署 —— 在线微调、多任务泛化

### 当前性能基线

| 指标 | 实测值 | 目标 | 状态 |
|------|--------|------|------|
| 端到端延迟 (tracker→mapper) | p50: 8.1ms, p99: 30.4ms | <50ms | 达标 |
| IK 求解耗时 | mean: 26ms, p99: 32ms | <33ms (30Hz) | 达标 |
| 时间同步误差 | 687us | <5ms | 达标 |
| 录制有效帧率 (mock) | ~22 Hz | 30 Hz | 受限于单进程 executor |
| 仿真管道 (MuJoCo) | 已验证 | — | mock_tracker → mapper → safety(mujoco) → recorder 全链路贯通 |

### Stage 2 工具（`stage_2/`）

`stage_2/` 目录包含第二阶段的数据管道工具，不应污染根目录或 `stage_1/`。

- `lerobot_v3_converter.py` — LeRobot v3.0 Parquet 格式转换器，入口函数为 `convert_directory(input_dir, output_dir, fps=30, task_yaml=None)`。也可通过 `convert_to_lerobot --v3` 命令行调用。
- `quality_filters.py` — 9 项数据质量检查（完整性/运动学/语义/时间戳），`analyze_episode()` 返回 `QualityReport`
- `task_manager.py` — 数据采集任务定义与 LeRobot `tasks.parquet` 导出
- `time_sync_check.py` — 时间同步精度评估（ROS2 clock vs time.time()）
- `quality_check.py` — 录制 mock episode + 质量分析（端到端测试用）
- `ik_diagnostic.py` — IK 收敛性诊断
- `act_research.py` — ACT 基线研究工具：批量 mock 录制（`-n N`）、v3 转换、格式验证。端到端管道验证用。
- `train_act.py` — ACT 正式训练脚本：支持 GPU/CPU、cosine LR schedule、定期 checkpoint、CLI 参数。用法：`python3 train_act.py --steps 10000 --batch-size 32`
- `act_training_plan.md` — ACT 训练方案文档：数据格式、超参数（chunk_size=100）、GPU 资源预估、从录制到训练的完整流程
- `quality_report.md`、`simulation_research.md` — 分析报告
- `simulation/` — MuJoCo 仿真环境：
  - `xarm6.xml` — 精确 MJCF 模型（14 bodies，FK 误差 0mm）
  - `mujoco_interface.py` — `MujocoRobotInterface`（RobotInterface ABC 实现）
  - `keyboard_teleop.py` — **键盘遥操作**：WASD 移动、QE 升降、IJKLUO 旋转、空格夹爪
  - `viewer_node.py` — **3D 可视化**：订阅 /teleop/state/joints，MuJoCo 窗口渲染
  - `teleop_sim.launch.py` / `teleop_sim_keyboard.launch.py` — 仿真管道启动文件
- `tasks/` — 任务定义 YAML 示例

### ACT 数据管道快速参考

```bash
# 生成 50-episode mock 数据集 + 转换为 LeRobot v3.0：
bash act_check.sh 5.0 -n 50 -o data/act_dataset_50

# 验证数据集：
python3 -c "
import pandas as pd
df = pd.read_parquet('data/act_dataset_50/data/chunk-000/file-000.parquet')
print(f'{df.episode_index.nunique()} eps, {len(df)} frames')
"

# 小规模 CPU 训练验证（确认 LeRobot ACT 模块可用）：
python3 -c "
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
cfg = ACTConfig(
    chunk_size=100, n_action_steps=100, n_obs_steps=1,
    input_features={'observation.environment_state': PolicyFeature(shape=[6], type=FeatureType.ENV)},
    output_features={'action': PolicyFeature(shape=[6], type=FeatureType.ACTION)},
    dim_model=128, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
    dim_feedforward=512, dropout=0.1, use_vae=False,
)
model = ACTPolicy(cfg)
print(f'ACT model: {sum(p.numel() for p in model.parameters()):,} parameters')
"
```

### 仿真遥操作（键盘控制 + 3D 可视化）

容器已启用 GPU 透传和 X11 转发，可直接运行 MuJoCo 可视化。

```bash
# 三终端启动（都在容器内 make shell）：
# 终端 1：控制管道
ros2 launch launch teleop_sim_keyboard.launch.py

# 终端 2：键盘遥操作
python3 stage_2/simulation/keyboard_teleop.py

# 终端 3：3D 可视化
python3 stage_2/simulation/viewer_node.py
```

键盘控制：W/S前后、A/D左右、Q/E升降、I/K俯仰、J/L横滚、U/O偏航、空格夹爪、R复位。

录制：`ros2 service call /recorder/start std_srvs/srv/Trigger` → 操作 → `/recorder/stop`

### 诊断工具（`stage_1/tools/`）

- `latency_profiler.py` — 端到端延迟分析（支持 `--standalone` 内联启动全管道）

## 测试

```bash
# 纯 Python 模块（宿主机运行，无需 Docker）：
PYTHONPATH=/workspace/umi python3 -m pytest stage_1/tests/test_fk.py stage_1/tests/test_ik.py \
       stage_1/tests/test_jacobian.py stage_1/tests/test_mock_robot.py \
       stage_1/tests/test_hdf5_writer.py stage_1/tests/test_lerobot_converter.py -v

# 完整测试套件（含 ROS2 集成测试）—— 容器内运行：
make shell cmd="bash -c '
  sudo pip install -q --upgrade \"setuptools>=65,<80\" && \
  sudo pip install -q \"anyio<4\" && \
  cd /ros2_ws && colcon build && \
  source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  sudo bash -c \"echo \\\"# no-op\\\" > /opt/ros/humble/lib/python3.10/site-packages/launch_testing_ros_pytest_entrypoint.py\" && \
  python3 -m pytest /workspace/umi/stage_1/tests/ -v -p no:launch_testing -p no:launch_testing_ros'"

# 单个测试：
make shell cmd="bash -c 'sudo pip install -q --upgrade \"setuptools>=65,<80\" && cd /ros2_ws && colcon build && source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && python3 -m pytest /workspace/umi/stage_1/tests/test_ik.py::TestInverseKinematics::test_small_perturbation_recovery -v'"

# 通过 colcon 运行（ROS2 方式）：
make shell cmd="bash -c 'sudo pip install -q --upgrade \"setuptools>=65,<80\" && colcon test --packages-select teleop_bridge recorder safety'"
```

## 代码检查与格式化

```bash
# 在容器内执行：
black .          # 格式化 Python 代码
ruff check .     # 静态检查 Python 代码
```

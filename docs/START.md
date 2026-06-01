# Stage 1 启动指南

本文档详细介绍如何从零开始启动 Stage 1 的遥操作管道，覆盖 Docker 环境准备、ROS2 编译、测试验证，以及三种启动模式（Mock、真实硬件、仅录制）的完整流程。

## 目录

- [前置条件](#前置条件)
- [快速开始（首次使用）](#快速开始首次使用)
- [详细步骤](#详细步骤)
  - [1. 构建 Docker 镜像](#1-构建-docker-镜像)
  - [2. 进入容器](#2-进入容器)
  - [3. 编译 ROS2 包](#3-编译-ros2-包)
  - [4. 运行测试](#4-运行测试)
  - [5. 启动管道](#5-启动管道)
- [三种启动模式](#三种启动模式)
  - [Mock 模式（无硬件）](#mock-模式无硬件)
  - [真实硬件模式（xArm6 + Quest3）](#真实硬件模式xarm6--quest3)
  - [仅录制模式](#仅录制模式)
- [录制工作流](#录制工作流)
- [逐节点手动启动](#逐节点手动启动)
- [常用命令速查](#常用命令速查)
- [故障排查](#故障排查)

## 前置条件

- **宿主机**：Ubuntu 22.04（或其他支持 Docker 的 Linux 发行版）
- **Docker**：已安装 Docker Engine 和 Docker Compose（`docker compose` 子命令可用）
- **GPU**（可选）：如需容器内 GPU 加速，安装 `nvidia-container-toolkit` 并取消 `docker-compose.yml` 中 GPU 相关配置的注释
- **xArm6**（可选）：真实硬件模式需要机械臂在同一网络中
- **Quest3**（可选）：真实硬件模式需要 Quest3 运行 Hand Tracking Streamer App

## 快速开始（首次使用）

以下命令在**宿主机**上执行，从构建到运行完整端到端 Mock 管道：

```bash
# 1. 构建 Docker 镜像（首次约 10-15 分钟，仅需一次）
make build

# 2. 编译 + 测试 + 启动 Mock 管道（一步完成）
make shell cmd="bash -c '
  sudo pip install -q --upgrade \"setuptools>=65,<80\" && \
  cd /ros2_ws && colcon build && \
  source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  pytest /workspace/umi/stage_1/tests/ -v -p no:launch_testing -p no:launch_testing_ros && \
  ros2 launch launch teleop_mock.launch.py'"
```

如果一切正常，你会看到：
- 36/36 tests passed
- 管道启动日志：Mock hand tracker、hand mapper、safety guardian、recorder 四个节点依次就绪

## 详细步骤

### 1. 构建 Docker 镜像

```bash
# 完整重新构建（首次或 Dockerfile 修改后）
make build

# 增量构建（利用 Docker 层缓存，仅重新构建变更的层）
make rebuild
```

构建产物：一个名为 `umi-dev` 的 Docker 镜像，包含 ROS2 Humble、Pinocchio、PyTorch、LeRobot、hand-tracking-sdk、xarm-python-sdk 等全部依赖。

### 2. 进入容器

容器启动方式有两种：

```bash
# 方式 A：交互式 shell（最常用）
make shell
# 退出：exit 或 Ctrl+D，容器自动删除

# 方式 B：后台运行（适合需要长时间保持服务运行）
make up          # 后台启动
make exec        # 进入后台容器的 shell
make exec cmd="ros2 topic list"  # 在后台容器执行指定命令
make down        # 停止并删除后台容器
```

两者的关键区别：
- `make shell`：每次启动新容器，退出后删除（`--rm`），状态不保留
- `make up`：容器持续运行，可以多次 `make exec` 进入同一个容器

**重要**：`make shell cmd="..."` 每次创建新容器，之前 `pip install` 和 `colcon build` 的产物不保留。如果有多个操作需要按顺序执行，用 `bash -c '...'` 串联在一起。

### 3. 编译 ROS2 包

进入容器后：

```bash
# 1. 降级 setuptools（colcon-core 兼容性要求）
sudo pip install --upgrade "setuptools>=65,<80"

# 2. 编译所有 ROS2 包
cd /ros2_ws && colcon build

# 3. 加载编译产物
source /ros2_ws/install/setup.bash
```

为什么需要降级 setuptools？Docker 镜像内的 setuptools 80.x 与 colcon-core 不兼容，会导致 `colcon build` 失败。`setuptools>=65,<80` 是已知兼容版本。

关于 `--symlink-install`：**不要使用这个标志**。因为 `package_dir` 配置使用了 `"."`（当前目录映射），`--symlink-install` 会在 Docker overlay 文件系统上触发 `OSError: Invalid argument`。不加 `--symlink-install` 时 colcon 会将编译产物复制到 install 目录，速度略慢但稳定。

entrypoint.sh 已在容器启动时自动将 `stage_1/<pkg>/` 目录符号链接到 `/ros2_ws/src/`，所以源码修改后只需重新 `colcon build` 即可。

**注意 CycloneDDS**：`docker-compose.yml` 配置了 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，但该包未默认安装。如果运行 ros2 命令时报 `rmw_cyclonedds_cpp not found`，有两种解法：

```bash
# 方案 A：切换到 FastRTPS（临时）
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 方案 B：安装 CycloneDDS（持久）
sudo apt-get update && sudo apt-get install -y ros-humble-rmw-cyclonedds-cpp
```

### 4. 运行测试

验证所有组件正常工作：

```bash
# 完整测试套件（容器内）
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
pytest /workspace/umi/stage_1/tests/ -v -p no:launch_testing -p no:launch_testing_ros

# 仅纯 Python 测试（宿主机即可运行，无需 ROS2）
PYTHONPATH=/workspace/umi pytest stage_1/tests/test_fk.py \
    stage_1/tests/test_ik.py stage_1/tests/test_jacobian.py \
    stage_1/tests/test_mock_robot.py stage_1/tests/test_hdf5_writer.py \
    stage_1/tests/test_lerobot_converter.py -v

# 单个测试文件
pytest stage_1/tests/test_ik.py -v

# 单个测试用例
pytest stage_1/tests/test_ik.py::TestInverseKinematics::test_small_perturbation_recovery -v
```

关于 `-p no:launch_testing -p no:launch_testing_ros`：项目的 `launch/` ROS2 包与 ROS2 核心的 `launch` Python 模块同名。当工作空间的 PYTHONPATH 覆盖系统路径时，ROS2 的 pytest 插件会因找不到 `launch.actions` 而崩溃。`-p no:launch_testing` 禁用这些插件。

如果遇到 `PluginValidationError: unknown hook 'pytest_launch_collect_makemodule'`，还需要：

```bash
sudo bash -c 'echo "# no-op" > /opt/ros/humble/lib/python3.10/site-packages/launch_testing_ros_pytest_entrypoint.py'
```

### 5. 启动管道

必须先在容器内 `source /ros2_ws/install/setup.bash`（entrypoint.sh 已将其写入 `~/.bashrc`，新终端自动生效）。

```bash
# Mock 模式
ros2 launch launch teleop_mock.launch.py

# Mock 模式 + 自定义录制目录
ros2 launch launch teleop_mock.launch.py output_dir:=/tmp/my_recordings

# Mock 模式 + 相机
ros2 launch launch teleop_mock.launch.py enable_cameras:=true

# 真实硬件模式
ros2 launch launch teleop_real.launch.py robot_ip:=192.168.1.100

# 真实硬件模式 + 相机
ros2 launch launch teleop_real.launch.py robot_ip:=192.168.1.100 enable_cameras:=true
```

启动后，打开另一个终端查看 topic 确认数据在流动：

```bash
# 在另一个终端进入同一个容器
make exec
# 或重新进入一个新容器
make shell

# 查看所有活跃 topic
ros2 topic list

# 查看关节指令数据
ros2 topic echo /teleop/command/joints

# 查看安全状态
ros2 topic echo /safety/status
```

## 三种启动模式

### Mock 模式（无硬件）

不需要 Quest3、不需要 xArm6、不需要相机。适合开发和调试。

节点拓扑：

```
mock_hand_tracker ──→ hand_mapper ──→ safety_guardian (mock) ──→ recorder
                                             │
                                        MockRobotInterface (内存仿真)
```

```bash
ros2 launch launch teleop_mock.launch.py
```

启动后你会看到：
- `[mock_hand_tracker]` Mock hand tracker started
- `[hand_mapper]` Hand mapper started (hand=right, scale=3.0)
- `[safety_guardian]` Safety guardian: mock mode
- `[recorder]` Recorder ready

hand_mapper 会周期性地输出 IK 失败警告，这是正常的——Lissajous 轨迹的某些端点位置超出了机械臂工作空间（典型 error 约 0.05-0.15 rad）。

### 真实硬件模式（xArm6 + Quest3）

需要：
1. Quest3 运行 Hand Tracking Streamer App，配置 UDP 发送到宿主机 IP:12345
2. xArm6 在同一网络中，已知 IP 地址
3. （可选）USB 相机连接到宿主机

```bash
# 启动
ros2 launch launch teleop_real.launch.py robot_ip:=192.168.1.100

# 如果有标定文件，在 hand_mapper 中使用
# 手动编辑 launch 文件或通过参数覆盖：
ros2 launch launch teleop_real.launch.py robot_ip:=192.168.1.100
```

节点拓扑：

```
Quest3 (UDP) ──→ hand_tracking_node ──→ hand_mapper ──→ safety_guardian (xarm6) ──→ recorder
                                                               │
                                                          XArm6Interface
```

### 仅录制模式

其他节点已经单独启动，只需要启动 recorder：

```bash
ros2 launch launch record_only.launch.py output_dir:=/tmp/my_recordings
```

节点拓扑：

```
(外部已运行的命令/状态 topics) ──→ recorder
```

## 录制工作流

启动管道后，通过 ROS2 service 控制录制：

```bash
# 开始录制
ros2 service call /recorder/start std_srvs/srv/Trigger

# 操作机械臂进行演示...

# 停止录制
ros2 service call /recorder/stop std_srvs/srv/Trigger
```

Recorder 会将数据写入 `output_dir` 参数指定的目录（默认 `~/umi_recordings`）：

```
~/umi_recordings/
├── episode_000000.h5
├── episode_000001.h5
└── ...
```

每个 episode 的 HDF5 文件包含：

| HDF5 Key | 含义 | 维度 |
|----------|------|------|
| `joint_command/position` | IK 目标关节角 | (N, 6) |
| `joint_state/position` | 实际关节角（经安全检验） | (N, 6) |
| `joint_state/velocity` | 实际关节速度 | (N, 6) |
| `gripper/command` | 夹爪指令 | (N,) |
| `gripper/state` | 实际夹爪开度 | (N,) |
| `sensors/camera/rgb` | RGB 图像（可选，需启用） | (N, H, W, 3) |
| `sensors/camera/depth` | 深度图（可选，需启用） | (N, H, W) |

导出为 LeRobot 格式：

```bash
# 转换单个文件
convert_to_lerobot --input episode_000000.h5 --output /path/to/lerobot_dataset/

# 批量转换整个录制目录，生成 features.json 元数据
convert_to_lerobot --input ~/umi_recordings --output ~/lerobot_dataset --features
```

## 逐节点手动启动

有时需要单独启动某个节点进行调试，而不是通过 launch 文件一键启动：

```bash
# 终端 1：Mock 手部追踪
ros2 run teleop_bridge mock_hand_tracker --ros-args -p frequency:=60.0

# 终端 2：手部映射
ros2 run teleop_bridge hand_mapper --ros-args -p hand:=right -p scale:=3.0 -p lowpass_alpha:=0.3

# 终端 3：安全守护
ros2 run safety safety_guardian --ros-args -p robot_mode:=mock

# 终端 4：录制器
ros2 run recorder recorder --ros-args -p output_dir:=/tmp/my_recordings
```

真实硬件时：

```bash
# 替换终端 1
ros2 run teleop_bridge hand_tracking_node --ros-args -p transport:=udp -p host:=0.0.0.0 -p port:=12345 -p hand:=right

# 替换终端 3
ros2 run safety safety_guardian --ros-args -p robot_mode:=xarm6 -p xarm6_ip:=192.168.1.100
```

相机节点：

```bash
ros2 run perception camera_node --ros-args -p device_id:=0 -p width:=640 -p height:=480 -p fps:=30.0
```

## 常用命令速查

```bash
# === 容器管理 ===
make build                      # 构建镜像（完整）
make rebuild                    # 构建镜像（利用缓存）
make shell                      # 进入交互式 shell
make shell cmd="pytest"         # 在容器内执行命令
make up                         # 后台运行
make down                       # 停止后台容器
make exec cmd="ros2 topic list" # 在后台容器执行命令
make clean                      # 清理构建缓存

# === ROS2 编译（容器内） ===
sudo pip install --upgrade "setuptools>=65,<80"
cd /ros2_ws && colcon build
source /ros2_ws/install/setup.bash

# === ROS2 调试（容器内） ===
ros2 topic list                  # 列出所有活跃 topic
ros2 topic echo /topic           # 实时查看 topic 数据
ros2 topic hz /topic             # 查看 topic 发布频率
ros2 node list                   # 列出所有活跃节点
ros2 node info /node_name        # 查看节点详情
ros2 service list                # 列出所有 service
ros2 service call /srv_name type # 调用 service

# === 录制控制 ===
ros2 service call /recorder/start std_srvs/srv/Trigger
ros2 service call /recorder/stop std_srvs/srv/Trigger

# === 安全控制 ===
ros2 service call /safety/reset std_srvs/srv/Trigger
ros2 service call /safety/enable std_srvs/srv/Trigger
ros2 service call /safety/disable std_srvs/srv/Trigger

# === LeRobot 导出 ===
convert_to_lerobot --input ~/umi_recordings --output ~/lerobot_dataset --features

# === 代码质量 ===
black .          # 格式化
ruff check .     # 静态检查
```

## 故障排查

### `colcon build` 输出 "0 packages finished"

entrypoint.sh 的符号链接可能指向了空的根目录壳体（只有 `__init__.py`，没有 `package.xml`），而非 `stage_1/<pkg>/`。

```bash
# 检查符号链接
ls -la /ros2_ws/src/

# 手动修复（已在 rebuild 后自动生效）
rm -f /ros2_ws/src/teleop_bridge /ros2_ws/src/perception /ros2_ws/src/safety /ros2_ws/src/recorder /ros2_ws/src/launch
ln -sf /workspace/umi/stage_1/teleop_bridge /ros2_ws/src/teleop_bridge
ln -sf /workspace/umi/stage_1/perception /ros2_ws/src/perception
ln -sf /workspace/umi/stage_1/safety /ros2_ws/src/safety
ln -sf /workspace/umi/stage_1/recorder /ros2_ws/src/recorder
ln -sf /workspace/umi/stage_1/launch /ros2_ws/src/launch
cd /ros2_ws && colcon build
```

注意：**需要 `make rebuild` 来让 entrypoint.sh 的修复永久生效**，因为 entrypoint.sh 在镜像构建时被 COPY 到镜像内，不会被 volume 挂载覆盖。

### `colcon build` 报 `OSError: [Errno 22] Invalid argument`

使用了 `--symlink-install` 标志。去掉这个标志，直接用 `colcon build`。

### `ros2` 命令报 `rmw_cyclonedds_cpp not found`

```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### pytest: `PluginValidationError: unknown hook 'pytest_launch_collect_makemodule'`

```bash
sudo bash -c 'echo "# no-op" > /opt/ros/humble/lib/python3.10/site-packages/launch_testing_ros_pytest_entrypoint.py'
pytest ... -p no:launch_testing -p no:launch_testing_ros
```

### pytest: `ModuleNotFoundError: No module named '_pytest.scope'`

`anyio` 最新版需要较新的 pytest，但系统自带 pytest 6.2.5：

```bash
sudo pip install "anyio<4"
```

### hand_mapper 不断输出 "IK failed"

这是正常的。Mock tracker 的轨迹范围超出 xArm6 工作空间（约 0.15m-0.7m 半径）。IK 求解器返回最佳近似解，位置误差通常 < 0.15m。如果真实硬件也有此问题，检查 `HandToRobotTransform` 标定参数是否合理。

### 录制后 HDF5 文件为空或没有数据

检查几点：
1. 录制时是否调用了 `/recorder/start` 和 `/recorder/stop` service
2. 录制路径是否有写入权限
3. 用 `h5ls -r episode_000000.h5` 检查文件内容

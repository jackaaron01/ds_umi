# Stage 1 — 硬件集成与遥操作回路

Stage 1 是 UMI 项目的第一阶段，实现 Quest3 VR 头显手部追踪到 xArm6 机械臂的实时遥操作控制，并同步录制多模态演示数据。全系统可在 mock 模式（无硬件）或真实硬件模式下运行。

## 目录

- [架构概览](#架构概览)
- [数据流管道](#数据流管道)
- [模块详解](#模块详解)
  - [kinematics/ — 运动学引擎](#kinematics--运动学引擎)
  - [robot_hal/ — 硬件抽象层](#robot_hal--硬件抽象层)
  - [teleop_bridge/ — 遥操作桥接](#teleop_bridge--遥操作桥接)
  - [safety/ — 安全守护](#safety--安全守护)
  - [recorder/ — 数据录制](#recorder--数据录制)
  - [perception/ — 相机驱动](#perception--相机驱动)
  - [launch/ — 启动编排](#launch--启动编排)
  - [tests/ — 测试](#tests--测试)
- [使用指南](#使用指南)
- [设计决策](#设计决策)

## 架构概览

### 分层设计

```
┌─────────────────────────────────────────────────┐
│  launch/                                        │  编排层
│  teleop_mock.launch.py / teleop_real.launch.py  │
└──────────┬──────────┬──────────┬────────────────┘
           │          │          │
    ┌──────▼──┐ ┌─────▼────┐ ┌──▼──────────┐
    │teleop   │ │ safety/  │ │ recorder/    │       ROS2 节点层
    │_bridge/ │ │          │ │              │       (ament_python)
    └────┬────┘ └────┬─────┘ └──────┬───────┘
         │           │              │
    ┌────▼───────────▼──────────────▼───────┐
    │  kinematics/   │   robot_hal/         │       纯 Python 层
    │  FK / IK /     │   RobotInterface     │       (无 ROS2 依赖)
    │  Jacobian      │   Mock / XArm6       │
    └────────────────┴──────────────────────┘
```

- **纯 Python 层**（`kinematics/`、`robot_hal/`）刻意不依赖 ROS2，可在任何环境测试
- **ROS2 节点层**（`teleop_bridge/`、`safety/`、`recorder/`、`perception/`）是 ament_python 包，通过 colcon 构建
- **编排层**（`launch/`）通过 ros2 launch 一键启动全管道

### 目录结构

```
stage_1/
├── kinematics/            # 运动学引擎（纯 Python）
│   ├── dh_params.py       #   xArm6 DH 参数表、关节限位、速度限位
│   ├── fk.py              #   正运动学（link_transforms, end_effector_pose）
│   ├── ik.py              #   逆运动学（Levenberg-Marquardt 阻尼最小二乘）
│   ├── jacobian.py        #   几何雅可比（解析形式）
│   └── utils.py           #   DH 变换、旋转矩阵/四元数/欧拉角互转、so3_log、pose_error
│
├── robot_hal/             # 硬件抽象层（纯 Python）
│   ├── __init__.py        #   RobotInterface ABC、JointState、GripperState 数据类
│   ├── mock_robot.py      #   MockRobotInterface（内存仿真，支持噪声注入/延迟模拟）
│   └── xarm6_interface.py #   XArm6Interface（真实 xArm6，通过 xarm-python-sdk）
│
├── teleop_bridge/         # 遥操作桥接（ROS2 包）
│   ├── hand_mapper.py     #   手部位姿 → IK → 关节/夹爪指令（低通滤波）
│   ├── mock_hand_tracker.py   #   Mock 手部追踪（Lissajous 合成轨迹）
│   ├── hand_tracking_node.py  #   Quest3 UDP 接收节点（hand-tracking-sdk 桥接）
│   ├── calibration.py     #   HandToRobotTransform（Quest3 → 机器人坐标系映射）
│   ├── calibrate.py       #   手眼标定工具（SVD 求解相似变换）
│   ├── setup.py           #   colcon 构建配置
│   └── package.xml        #   ROS2 包清单
│
├── safety/                # 安全守护（ROS2 包）
│   ├── safety_node.py     #   SafetyGuardian 节点（状态机、限位/速度/增量检查、急停）
│   ├── setup.py
│   └── package.xml
│
├── recorder/              # 数据录制（ROS2 包）
│   ├── hdf5_writer.py     #   HDF5Writer（线程安全、层级 key、GZIP 压缩）
│   ├── recorder_node.py   #   RecorderNode（ring buffer、service 控制、30Hz 刷新）
│   ├── lerobot_converter.py   #   LeRobot 格式转换（CLI + library）
│   ├── setup.py
│   └── package.xml
│
├── perception/            # 相机驱动（ROS2 包）
│   ├── camera_node.py     #   通用 USB 相机节点（OpenCV VideoCapture + 标定文件加载）
│   ├── setup.py
│   └── package.xml
│
├── launch/                # 启动编排（ROS2 包）
│   ├── teleop_mock.launch.py  #   Mock 模式（无硬件）
│   ├── teleop_real.launch.py  #   真实硬件模式（xArm6 + Quest3）
│   ├── record_only.launch.py  #   仅录制模式
│   ├── setup.py
│   └── package.xml
│
└── tests/                 # 测试
    ├── conftest.py            #   共享 fixture（dh_params, random_q, sample_configs）
    ├── test_fk.py             #   正运动学测试（8 项）
    ├── test_ik.py             #   逆运动学测试（收敛性、自洽性、扰动恢复）
    ├── test_jacobian.py       #   雅可比测试（与有限差分数值验证对比）
    ├── test_mock_robot.py     #   MockRobotInterface 测试（8 项）
    ├── test_hdf5_writer.py    #   HDF5Writer 测试（6 项）
    ├── test_lerobot_converter.py  #   LeRobot 转换测试
    ├── test_camera_node.py    #   相机节点测试
    ├── test_recorder_image.py #   录制器图像处理测试
    └── test_integration.py    #   端到端集成测试（mock pipeline）
```

## 数据流管道

```
                    ┌──────────────────┐
                    │  Quest3 或       │
                    │  mock_hand_tracker│
                    └───┬─────────┬────┘
                        │         │
              /hand/{hand}/wrist_pose  /hand/{hand}/keypoints
              (PoseStamped)            (Float32MultiArray)
                        │         │
                    ┌───▼─────────▼───┐
                    │   hand_mapper   │
                    │                 │
                    │  坐标变换       │
                    │  (Calibration)  │
                    │  低通滤波       │
                    │  IK 求解        │
                    └───┬─────────┬───┘
                        │         │
           /teleop/command/joints  /teleop/command/gripper
           (JointState)            (Float64)
                        │         │
                    ┌───▼─────────▼───┐
                    │ safety_guardian │
                    │                 │
                    │  关节限位检查   │
                    │  速度/增量检查  │
                    │  状态机转换     │
                    │  转发至 Robot   │
                    └───┬─────────┬───┘
                        │         │
              /teleop/state/joints  /teleop/state/gripper
              (JointState)          (Float64)
                        │         │
                    ┌───▼─────────▼───┐
                    │    recorder     │
                    │                 │
                    │  Ring Buffer    │
                    │  HDF5 Writer    │
                    │  (30 Hz flush)  │
                    └─────────────────┘
                              │
                    可选: /camera/rgb/image_raw
                    可选: /camera/depth/image_raw
                              │
                        ┌─────▼─────┐
                        │  camera   │
                        │  _node    │
                        └───────────┘
```

### ROS2 Topic 合约总表

| Topic | 类型 | 发布者 | 订阅者 | 说明 |
|-------|------|--------|--------|------|
| `/hand/{hand}/wrist_pose` | `PoseStamped` | mock_tracker / hand_tracking_node | hand_mapper | 手腕 6D 位姿 |
| `/hand/{hand}/keypoints` | `Float32MultiArray` | mock_tracker / hand_tracking_node | hand_mapper | 21 个手部关键点（×3 = 63 个 float） |
| `/teleop/command/joints` | `JointState` | hand_mapper | safety, recorder | IK 求解的 6 关节目标角 |
| `/teleop/command/gripper` | `Float64` | hand_mapper | safety, recorder | 夹爪目标开度 [0,1] |
| `/teleop/state/joints` | `JointState` | safety | recorder | 经安全检验后的实际关节角（含 velocity/effort） |
| `/teleop/state/gripper` | `Float64` | safety | recorder | 实际夹爪开度 |
| `/safety/status` | `String` | safety | (监控) | 安全状态：NORMAL / WARNING / EMERGENCY_STOP |
| `/camera/rgb/image_raw` | `Image` | camera_node | recorder (可选) | RGB 图像 |
| `/camera/depth/image_raw` | `Image` | camera_node | recorder (可选) | 深度图 |

### ROS2 Service 接口

| Service | 类型 | 节点 | 说明 |
|---------|------|------|------|
| `/recorder/start` | `Trigger` | recorder | 开始录制新 episode |
| `/recorder/stop` | `Trigger` | recorder | 停止当前 episode 并写入磁盘 |
| `/safety/reset` | `Trigger` | safety | 从 ESTOP 状态恢复到 NORMAL |
| `/safety/enable` | `Trigger` | safety | 启用指令转发 |
| `/safety/disable` | `Trigger` | safety | 禁用指令转发（软急停） |

## 模块详解

### kinematics/ — 运动学引擎

纯 Python 实现，零外部依赖（仅 NumPy）。xArm6 参数化，修改 `dh_params.py` 可适配其他机械臂。

#### DH 参数（`dh_params.py`）

xArm6 标准 DH 参数表，每行 `[a, alpha, d, theta_offset]`：

| Joint | a (m) | α (rad) | d (m) | θ offset (rad) |
|-------|-------|---------|-------|-----------------|
| 1 | 0.0 | 0.0 | 0.267 | 0.0 |
| 2 | 0.0 | -π/2 | 0.0 | -π/2 |
| 3 | 0.2895 | 0.0 | 0.0 | 0.0 |
| 4 | 0.0775 | -π/2 | 0.3425 | 0.0 |
| 5 | 0.0 | π/2 | 0.0 | 0.0 |
| 6 | 0.0 | -π/2 | 0.0975 | 0.0 |

同时定义了关节限位、速度限位（180°/s）和关节名称。

#### 正运动学（`fk.py`）

- `link_transforms(q)` — 计算从基座到各连杆的 4×4 齐次变换矩阵（返回 7 个，含基座）
- `forward_kinematics(q)` — 返回所有 7 个坐标系的位置 `(7,3)` 和旋转矩阵 `(7,3,3)`
- `end_effector_pose(q)` — 只返回末端执行器的旋转矩阵 `(3,3)` 和位置 `(3,)`

所有函数接受可选 `dh_params` 参数以支持不同机械臂。

#### 逆运动学（`ik.py`）

- `solve_ik(target_pose, q_init, ...)` — Levenberg-Marquardt 阻尼最小二乘法
  - 输入：4×4 齐次变换目标位姿
  - 可选初始猜测 `q_init`（默认为零位）
  - 收敛判据：位置误差 < 1mm 且旋转误差 < 0.001 rad
  - 带线搜索（α ∈ {1.0, 0.5, 0.25}），自适应阻尼 λ ∈ [0.01, 1.0]
  - 返回 `(q_solution, success, iterations, final_error)`

#### 几何雅可比（`jacobian.py`）

- `compute_jacobian(q)` — 6×6 几何雅可比
  - 旋转关节 i 列：`[z_i × (p_ee - p_i), z_i]^T`
  - 与有限差分数值验证的一致性在 `test_jacobian.py` 中检验

#### 工具函数（`utils.py`）

| 函数 | 功能 |
|------|------|
| `dh_transform(a, α, d, θ)` | 标准 DH 参数 → 4×4 变换矩阵 |
| `skew(v)` | 3 向量 → 3×3 反对称矩阵 |
| `rotation_matrix_to_euler(R)` | 旋转矩阵 → RPY 欧拉角 (XYZ fixed-axis) |
| `euler_to_rotation_matrix(rpy)` | RPY 欧拉角 → 旋转矩阵 |
| `rotation_matrix_to_quaternion(R)` | 旋转矩阵 → 四元数 `[w,x,y,z]` |
| `quaternion_to_rotation_matrix(q)` | 四元数 `[w,x,y,z]` → 旋转矩阵 |
| `so3_log(R)` | 旋转矩阵 → so(3) 轴角向量（对数映射） |
| `pose_error(T_cur, T_des)` | 两个位姿 → 6 维误差向量 `[pos_err, rot_err]`（世界坐标系下） |
| `pose_to_transform(pos, quat_xyzw)` | 位置 + 四元数 → 4×4 齐次变换 |

### robot_hal/ — 硬件抽象层

系统最重要的模块。所有上层代码只依赖 `RobotInterface` 抽象接口。

#### 抽象接口（`__init__.py`）

```python
class RobotInterface(ABC):
    def connect(self) -> bool: ...
    def disconnect(self) -> bool: ...
    def get_joint_state(self) -> JointState: ...
    def move_joints(self, positions, velocity, blocking) -> bool: ...
    def stop(self) -> bool: ...
    def get_gripper_state(self) -> GripperState: ...
    def move_gripper(self, position, blocking) -> bool: ...
    # 可选扩展:
    def move_cartesian(self, pose, velocity, blocking) -> bool: ...
    def get_cartesian_pose(self) -> np.ndarray: ...
```

数据类：

- `JointState(position, velocity, effort, name)` — 所有字段为 NumPy 数组
- `GripperState(position, effort)` — position ∈ [0,1]

#### MockRobotInterface（`mock_robot.py`）

纯内存实现，用于无硬件测试。特性：

- **blocking 移动**：基于距离和速度模拟运动时间
- **non-blocking 移动**：线性插值模拟，`get_joint_state()` 时推进
- **噪声注入**：`enable_noise(std_pos, std_vel)` 模拟传感器噪声
- **延迟模拟**：`enable_delay(seconds)` 模拟通信延迟
- **状态预设**：`set_joint_state(positions)` 用于测试特定配置

#### XArm6Interface（`xarm6_interface.py`）

真实 xArm6 驱动，惰性导入 `xarm-python-sdk`。

```python
robot = XArm6Interface(ip="192.168.1.100")
robot.connect()
robot.move_joints(target, velocity=0.5, blocking=True)
robot.disconnect()
```

支持位置模式下的关节空间和笛卡尔空间控制（SDK 的 `set_servo_angle` / `set_position`）。

### teleop_bridge/ — 遥操作桥接

#### HandToRobotTransform（`calibration.py`）

将 Quest3 追踪空间的手腕位姿映射到机器人基座坐标系：

```
p_robot = scale * R @ p_quest + offset
```

- 默认变换：Quest3 坐标系（+X 右, +Y 上, +Z 后）→ 机器人坐标系（+X 前, +Y 左, +Z 上）
- 默认缩放因子：3.0（手部运动 ×3 映射到大臂运动）
- 支持 `from_yaml()` 加载标定文件
- `mock_transform()`：无硬件测试用工厂方法（scale=1.0, offset=[0.6, 0, 0.3]）

#### hand_mapper 节点（`hand_mapper.py`）

核心遥操作节点。信号链：

1. 订阅 `PoseStamped` 手腕位姿 → 6D 目标位姿
2. 低通滤波（指数平滑，默认 α=0.3）
3. `HandToRobotTransform` 坐标系映射
4. 构建 4×4 齐次变换目标
5. `solve_ik()` 求解关节角
6. 发布 `JointState` 到 `/teleop/command/joints`

夹爪控制：

1. 订阅 21 个手部关键点（`Float32MultiArray`，63 个 float）
2. 计算拇指尖（landmark 4）与食指尖（landmark 8）距离
3. 线性映射到 [0,1]（close_thresh=0.015m, open_thresh=0.080m）
4. 发布 `Float64` 到 `/teleop/command/gripper`

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hand` | `"right"` | 追踪左手还是右手 |
| `scale` | `3.0` | 手部运动缩放因子 |
| `lowpass_alpha` | `0.3` | 低通滤波系数（越小越平滑，越大响应越快） |
| `calibration_file` | `""` | 标定 YAML 文件路径（空串则使用默认参数） |

#### mock_hand_tracker 节点（`mock_hand_tracker.py`）

以 60 Hz 发布合成手部数据（Lissajous 8 字形轨迹），用于无 Quest3 的测试。

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `frequency` | `60.0` | 发布频率 (Hz) |
| `amplitude_x` | `0.15` | X 方向振幅 (m) |
| `amplitude_y` | `0.10` | Y 方向振幅 (m) |
| `amplitude_z` | `0.10` | Z 方向振幅 (m) |
| `omega` | `0.5` | 角频率 |
| `offset_z` | `0.3` | Z 轴偏移 (m) |

夹爪开合由正弦波调制（模拟捏合-松开）。

#### hand_tracking_node（`hand_tracking_node.py`）

Quest3 手部追踪数据的 ROS2 桥接节点（需要 `hand-tracking-sdk`）：

- 支持 UDP/TCP Server/TCP Client 三种传输模式
- 后台线程接收 HTSClient 数据流
- ROS2 定时器（默认 60 Hz）发布最新数据到标准 topic
- 错误容忍策略（`ErrorPolicy.TOLERANT`），单帧异常不中断

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `transport` | `"udp"` | 传输模式 |
| `host` | `"0.0.0.0"` | 监听地址 |
| `port` | `12345` | 端口号 |
| `hand` | `"right"` | 追踪手 |
| `publish_interval` | `1/60` | 发布间隔 (s) |

#### calibrate（`calibrate.py`）

手眼标定工具（独立脚本，无 ROS2 依赖）：

```bash
# 从 JSON 标定数据求解
python calibrate.py --input pairs.json --output calibration.yaml

# 交互式输入
python calibrate.py --interactive --output calibration.yaml
```

- 使用 Umeyama SVD 方法求解相似变换（旋转 + 缩放 + 平移）
- 最少 3 对配准点，输出 YAML 文件可直接被 `HandToRobotTransform.from_yaml()` 加载

### safety/ — 安全守护

#### SafetyGuardian 节点（`safety_node.py`）

验证遥操作指令并在安全范围内转发给机器人。

**状态机**（三态）：

```
NORMAL ←→ WARNING → EMERGENCY_STOP
  ↑                    │
  └────── reset ───────┘
```

- `NORMAL`：正常转发指令
- `WARNING`：触发了安全条件但仍在可修复范围内。连续 3 次 WARNING 自动升级为 ESTOP。Reset 清零计数器
- `EMERGENCY_STOP`：停止所有运动。需通过 `/safety/reset` 服务恢复

**安全检查**（每次控制循环执行）：

| 检查项 | 参数 | 行为 |
|--------|------|------|
| 关节限位 | `XARM6_JOINT_LIMITS` | clamp 越界值，触发 WARNING |
| 单步增量 | `position_delta_limit` (默认 0.3 rad) | clamp 超限值，触发 WARNING |
| 速度限制 | `velocity_limit` (默认 π rad/s) | 当前仅为参数记录 |

**控制循环**（默认 60 Hz）：

1. 读取机器人状态 → 发布 `/teleop/state/joints`、`/teleop/state/gripper`
2. 消费 pending command → 安全检查 → `robot.move_joints()`（non-blocking）
3. 消费 pending gripper command → `robot.move_gripper()`
4. 发布 `/safety/status`

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `robot_mode` | `"mock"` | `"mock"` 或 `"xarm6"` |
| `xarm6_ip` | `"192.168.1.100"` | xArm6 控制器 IP |
| `velocity_limit` | `π` | 速度限制 (rad/s) |
| `position_delta_limit` | `0.3` | 单步增量限制 (rad) |
| `control_rate` | `60.0` | 控制循环频率 (Hz) |

### recorder/ — 数据录制

#### HDF5Writer（`hdf5_writer.py`）

线程安全的 HDF5 录制器，核心设计：

- **层级 key 结构**：`group/subgroup/dataset_name`（含 `/` 的 key 自动创建 HDF5 group，无 `/` 的 key 静默跳过）
- **可扩展数据集**：`maxshape=(None, ...)`，每次 `write_step()` 调用 `dset.resize()`
- **GZIP 压缩**：compression=4（速度和文件大小的平衡点）
- **Episode 管理**：`start_episode()` / `end_episode()` 生命周期
- **元数据**：episode 级 attrs（`num_steps`、`date`、自定义 key-value）

写入格式：

| HDF5 Key | 数据类型 | 维度 |
|----------|---------|------|
| `joint_command/position` | float64 | (N, 6) |
| `joint_state/position` | float64 | (N, 6) |
| `joint_state/velocity` | float64 | (N, 6) |
| `gripper/command` | float64 | (N,) |
| `gripper/state` | float64 | (N,) |
| `sensors/camera/rgb` | uint8 | (N, H, W, 3) |
| `sensors/camera/depth` | uint8/uint16 | (N, H, W) |

#### RecorderNode（`recorder_node.py`）

ROS2 录制节点：

- **Ring Buffer 架构**：每个 topic 独立 `deque`（默认 maxlen=100，图像 maxlen=5），持续缓冲最新数据
- **30 Hz 刷新线程**：从各 buffer 取最新条目（by timestamp），合并写入同一 step
- **Service 控制**：`/recorder/start` 和 `/recorder/stop` 管理录制生命周期
- **可选图像录制**：通过 `enable_image_recording` 参数启用

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `~/umi_recordings` | HDF5 输出目录 |
| `auto_start` | `False` | 是否启动后自动开始录制 |
| `enable_image_recording` | `False` | 是否订阅并录制相机图像 |
| `image_topic_rgb` | `/camera/rgb/image_raw` | RGB 图像 topic |
| `image_topic_depth` | `/camera/depth/image_raw` | 深度图 topic |

#### LeRobot 转换器（`lerobot_converter.py`）

将 UMI Stage 1 格式转换为 [LeRobot](https://github.com/huggingface/lerobot) 标准格式：

```bash
# 转换单个文件
convert_to_lerobot --input episode_000000.h5 --output /path/to/lerobot_dataset/

# 批量转换整个目录并生成 features.json
convert_to_lerobot --input /tmp/umi_recordings --output /tmp/lerobot_export --features
```

Key 映射：

| UMI Stage 1 | LeRobot |
|-------------|---------|
| `joint_command/position` | `action/joint_position` |
| `joint_state/position` | `observation/joint_position` |
| `joint_state/velocity` | `observation/joint_velocity` |
| `gripper/command` | `action/gripper` |
| `gripper/state` | `observation/gripper` |
| `sensors/camera/rgb` | `observation/images/camera_rgb` |

输出结构：

```
lerobot_dataset/
├── data/
│   └── episode_000000.h5
├── meta/
│   ├── episodes.parquet    (episode_index + length)
│   └── features.json       (可选，--features 时生成)
```

也可作为 Python 库使用：

```python
from stage_1.recorder.lerobot_converter import convert_episode, write_episodes_metadata
convert_episode("episode_000000.h5", "/output/", episode_index=0)
write_episodes_metadata("/output/", [(0, 100)])
```

### perception/ — 相机驱动

#### CameraNode（`camera_node.py`）

通用 USB 相机 ROS2 驱动：

- 基于 OpenCV `VideoCapture`，支持任意 `/dev/videoN` 设备
- 发布 `sensor_msgs/Image`（BGR→RGB 转换）和可选 `sensor_msgs/CameraInfo`
- 加载 YAML 标定文件（支持 camera_matrix、distortion、rectification、projection matrix）

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `device_id` | `0` | 相机设备 ID |
| `width` | `640` | 图像宽度 |
| `height` | `480` | 图像高度 |
| `fps` | `30.0` | 帧率 |
| `topic_name` | `/camera/rgb/image_raw` | 发布 topic 名称 |
| `calibration_file` | `""` | YAML 标定文件路径 |

### launch/ — 启动编排

三种预设启动配置：

#### teleop_mock.launch.py — Mock 模式

无硬件全系统测试。启动节点：

```
mock_hand_tracker → hand_mapper → safety_guardian (mock) → recorder
                                                         → camera_node (可选)
```

```bash
ros2 launch launch teleop_mock.launch.py
ros2 launch launch teleop_mock.launch.py output_dir:=/tmp/my_recordings enable_cameras:=true
```

#### teleop_real.launch.py — 真实硬件模式

Quest3 + xArm6 生产环境。启动节点：

```
hand_tracking_node → hand_mapper → safety_guardian (xarm6) → recorder
                                                            → camera_node (可选)
```

```bash
ros2 launch launch teleop_real.launch.py robot_ip:=192.168.1.100
```

#### record_only.launch.py — 仅录制模式

在其他节点已运行的场景下单独启动录制器。

```bash
ros2 launch launch record_only.launch.py output_dir:=/tmp/my_recordings
```

### tests/ — 测试

36 项测试，覆盖从底层运动学到端到端管道。

#### 测试组织

| 文件 | 数量 | 类型 | 依赖 |
|------|------|------|------|
| `test_fk.py` | 8 | 单元测试 | NumPy |
| `test_ik.py` | 10+ | 单元测试 | NumPy |
| `test_jacobian.py` | 6 | 单元测试 | NumPy |
| `test_mock_robot.py` | 8 | 单元测试 | NumPy |
| `test_hdf5_writer.py` | 6 | 单元测试 | h5py |
| `test_lerobot_converter.py` | 2 | 单元测试 | h5py, pandas |
| `test_camera_node.py` | 1+ | 集成测试 | ROS2 |
| `test_recorder_image.py` | 1+ | 集成测试 | ROS2 |
| `test_integration.py` | 1 | 端到端测试 | ROS2, h5py |

#### 关键测试说明

- **IK 收敛性**：零位配置、非零配置、随机配置的 FK→IK 自洽性（正向-逆向循环）
- **IK 小扰动恢复**：随机偏移 0.03 rad 后 IK 能否恢复到原末端位姿
- **雅可比验证**：与有限差分数值雅可比比较（中心差分，h=1e-6），精度 1e-4
- **HDF5 层级 key**：验证含 `/` 的 key 创建嵌套 group，无 `/` 的 key 被跳过
- **端到端集成**：mock_hand_tracker → hand_mapper → safety → recorder，运行 3 秒验证 HDF5 产出（>20 steps，含 joint_command 和 joint_state）

## 使用指南

### 环境准备

所有开发在 Docker 容器中进行（镜像：`osrf/ros:humble-desktop-full`，Ubuntu 22.04，ROS2 Humble）。

```bash
# 1. 构建镜像（首次或 Dockerfile 变更后）
make build

# 2. 进入容器
make shell

# 容器内：
# 3. 降级 setuptools（兼容 colcon-core）
sudo pip install --upgrade "setuptools>=65,<80"

# 4. 编译 ROS2 包
cd /ros2_ws && colcon build

# 5. Source 编译产物
source install/setup.bash
# （entrypoint 已将 source 写入 ~/.bashrc，新 shell 自动执行）
```

### 运行测试

```bash
# 纯 Python 模块（宿主机即可运行）：
PYTHONPATH=/workspace/umi pytest stage_1/tests/test_fk.py stage_1/tests/test_ik.py \
    stage_1/tests/test_jacobian.py stage_1/tests/test_mock_robot.py \
    stage_1/tests/test_hdf5_writer.py stage_1/tests/test_lerobot_converter.py -v

# 完整测试套件（容器内）：
make shell cmd="bash -c '\
  sudo pip install -q --upgrade \"setuptools>=65,<80\" && \
  cd /ros2_ws && colcon build && \
  source /opt/ros/humble/setup.bash && source install/setup.bash && \
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  pytest /workspace/umi/stage_1/tests/ -v -p no:launch_testing -p no:launch_testing_ros'"
```

### 运行管道

```bash
# Mock 模式（无硬件测试全管道）：
ros2 launch launch teleop_mock.launch.py

# 在另一个终端开始录制：
ros2 service call /recorder/start std_srvs/srv/Trigger

# 停止录制：
ros2 service call /recorder/stop std_srvs/srv/Trigger

# 真实硬件模式：
ros2 launch launch teleop_real.launch.py robot_ip:=192.168.1.100
```

### 手眼标定

```bash
# 1. 采集配准点（将末端执行器移动到已知位置，记录 Quest3 手腕坐标）
#    保存为 JSON:
#    [{"quest": [0.1, 0.2, 0.3], "robot": [0.4, 0.0, 0.3]}, ...]

# 2. 求解标定参数
python -m stage_1.teleop_bridge.calibrate --input pairs.json --output calibration.yaml

# 3. 使用标定结果启动
ros2 launch launch teleop_real.launch.py \
    robot_ip:=192.168.1.100
#   然后在 hand_mapper 节点中设置 calibration_file 参数
```

### 导出 LeRobot 数据集

```bash
# 将录制目录转为 LeRobot 格式
convert_to_lerobot --input ~/umi_recordings --output ~/lerobot_dataset --features
```

## 设计决策

### 为什么运动学不依赖 ROS2？

`kinematics/` 和 `robot_hal/` 是纯 Python 模块。这意味着：
- FK/IK 可以在 CI、Jupyter notebook、宿主机上直接测试，无需启动 Docker
- 框架不锁定 —— 未来迁移到 ROS2 Iron、ROS1，甚至一个自定义 Python 控制回路都不需要改动运动学代码
- Pinocchio 作为可选依赖（`robot_hal/__init__.py` 中通过 importorskip 使用），仅用于交叉验证

### 为什么使用 HAL 抽象层？

`RobotInterface` ABC 是系统中最核心的设计。所有上层代码（hand_mapper、safety、recorder）只依赖这个抽象接口。从 xArm6 换到 UR5 或仿真器，只需写一个新的 `XxxInterface` 实现 —— 其他代码零改动。

### 为什么 pose_error 使用世界坐标系旋转误差？

`pose_error()` 计算 `R_err = R_des @ R_cur^T`（世界坐标系），而非 `R_cur^T @ R_des`（本体坐标系）。这是因为 `compute_jacobian()` 的角速度列 `z_i` 在世界坐标系下表示，所以误差向量也必须在世界坐标系下，梯度方向才正确。这是 IK 收敛性的关键修正。

### 为什么 Recorder 使用 ring buffer + 30Hz 刷新？

ROS2 topic 是异步的 —— joint command 和 joint state 以不同频率到达。Recorder 为每个 topic 维护独立 ring buffer，在 30 Hz 固定频率下采样各 buffer 的最新数据，从而以统一时间戳生成时间对齐的 step。这避免了因回调顺序不确定导致的配对错误。

### 为什么 HDF5 key 必须含 `/`？

`HDF5Writer` 使用层级 key 约定（如 `joint_state/position`）来组织 HDF5 group 和 dataset。不含 `/` 的 key 被静默跳过，这是防止将错误格式数据写入文件的安全机制。如果你添加新的 data key，确保使用 `group/dataset` 格式。

### 为什么 Levenberg-Marquardt 初始阻尼是 0.01 而非 0.1？

初始阻尼 `λ=0.01` 是从默认位姿（零位）出发 IK 收敛的关键 —— 大阻尼会使第一步过于保守，梯度方向不足以引导求解器脱离局部区域。`λ` 在全过程中自适应：步长被接受时缩小（×0.5），被拒绝时增大（×2.0），范围 [0.01, 1.0]。

### 为什么 WARNING 连续 3 次才触发 ESTOP？

安全节点采用了"容忍性"设计：单次越限（如关节略超限位、网络抖动导致的瞬态速度尖峰）触发 WARNING 但不停止系统。连续 3 次 WARNING 才会升级为 ESTOP。这避免了因瞬时噪声频繁急停，同时确保持续违反安全条件能被及时拦截。

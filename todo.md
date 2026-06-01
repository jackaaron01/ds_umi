# Stage 1 TODO — 硬件集成与遥操作回路

> 规划依据：`planning/stage1_planning/stage1_v1.md`
> 目标：实现实时控制和同步录制，Quest3 → ROS2 → xArm6/Mock 闭环

---

## 已完成 ✅

- [x] **1A 项目骨架** — Docker 环境、ROS2 workspace、包结构、colcon 构建
- [x] **1B HAL 抽象层** — `RobotInterface` ABC、`JointState`/`GripperState`、`MockRobotInterface`、`XArm6Interface`
- [x] **1C 运动学** — FK（DH 参数）、IK（damped least squares）、Jacobian（解析 + 有限差分验证）
- [x] **1D 坐标变换** — `HandToRobotTransform`（Quest3 → Robot 坐标映射）
- [x] **1E 手部映射** — `hand_mapper` 节点（IK + 低通滤波 + 夹爪捏合检测）
- [x] **1F Mock 手部追踪** — `mock_hand_tracker` 节点（Lissajous 合成轨迹）
- [x] **1G 安全守护** — `safety_guardian` 节点（关节限位/速度/增量检查、状态机、急停）
- [x] **1H HDF5 录制** — `HDF5Writer`（线程安全、可扩展、分层 key）、`recorder` 节点（ring buffer、service 控制）
- [x] **1I LeRobot 转换** — key 映射、episode metadata (parquet)、features.json
- [x] **1J 相机驱动** — `camera_node`（USB 相机、标定文件加载）
- [x] **1K Launch 文件** — mock 模式、真实硬件模式、仅录制模式
- [x] **1L 测试** — FK/IK/Jacobian/MockRobot/HDF5/LeRobot 单元测试 + mock pipeline 集成测试

---

## 待完成 🔲

- [x] **2A Quest3 UDP 接收节点** — 用 `hand-tracking-sdk` 接收 Quest3 手部追踪 UDP 数据，发布 ROS2 topics（`/hand/{hand}/wrist_pose`、`/hand/{hand}/keypoints`）
- [x] **2B 手眼标定工具** — 采集配对位姿、求解相似变换（SVD）、输出标定 YAML，`HandToRobotTransform` 支持 `from_yaml()` 加载
- [x] **2C Recorder gripper state 修复** — 新增 `/teleop/state/gripper` 订阅，command 和 state 分离记录
- [x] **2D Safety 节点持久化 warning 升级** — 实现 `_warning_count` 计数器，连续 3 次 WARNING 自动触发 ESTOP，reset 时清零
- [x] **2E 端到端验证** — 36/36 测试通过（修复 IK pose_error 坐标系不匹配 + 集成测试校准参数）

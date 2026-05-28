好，先看整体系统架构，再逐子阶段展开。

---

**第一张图**是目标系统的软件分层结构——你要搭建的是什么。

---每个框都可以点击查看深度讲解。接下来是四个子阶段的详细任务拆解。

------

## 关于三个目标的对应设计

你提了三个核心约束，它们在架构上的体现是这样的：

**可复用性**：HAL 抽象层（1B）是关键。上层所有代码只依赖 `RobotInterface` 接口，不知道下面是 xArm6 还是 UR5 还是仿真机器人。后续换硬件时，只需写一个新的 `XxxInterface` 实现类，其余代码零改动。

**低成本/无实物采集**：`MockRobotInterface`（1B）是这个方向的入口。它让整个控制-记录流水线在没有任何硬件的情况下跑通，为后续接入仿真引擎（Isaac Gym / MuJoCo）或纯键鼠输入采集做好了接入点。Quest3 本身价格也比 Gello 机械臂便宜，这一步选择是合理的。

**以学为主**：每个子阶段的设计都刻意避开"直接复用"的路——FK/IK 手写、标定手跑、数据格式手写 HDF5。这不是在重复造轮子，而是把轮子拆开看清楚，之后用 Pinocchio 或者 LeRobot 工具的时候才知道它们在帮你省什么。

---

## 建议的代码仓库结构

```
umi_teleop/
├── robot_hal/              # HAL 抽象接口 + xArm6 实现 + Mock 实现
├── kinematics/             # 自研 FK / IK，不依赖 ROS
├── teleop_bridge/          # Quest3 → ROS2 桥接节点
├── perception/             # 相机节点 + 标定工具
├── safety/                 # SafetyGuardian 节点
├── recorder/               # HDF5 记录器 + LeRobot 格式转换
├── launch/                 # 全系统 launch 文件
└── tests/                  # 用 MockRobot 的无硬件测试
```

`kinematics/` 和 `robot_hal/` 是纯 Python 包，刻意不依赖 ROS2——这样单元测试在任何环境跑，未来迁移也不痛。

---

想先深入哪个子阶段？比如 1B 的 IK 推导、1C 的手眼标定数学、或者 1D 的 HDF5 数据结构设计，可以直接展开。

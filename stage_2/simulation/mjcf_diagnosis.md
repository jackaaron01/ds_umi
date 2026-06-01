# MJCF 模型精度诊断报告

> 2026-06-01 诊断 → 2026-06-01 修复完成
> 基于 `planning/tomorrow_2026-05-30.md` 任务 2 "MJCF 模型精度修复"

## 状态：✅ 已修复

**修复方案**：方案 B（两层 body 结构），commit `6e02630`。
- FK 误差：>1m → **0.000000m**
- 仿真管道测试：PASSED

## 问题（已解决）

~~当前 `stage_2/simulation/xarm6.xml` 的 MuJoCo FK 与手写 FK 输出不一致，仿真数据不可用于训练。~~

## 数值验证

```
xArm6 FK at q=0:    [0.440, -0.367, 0.267]
MJCF FK at q=0:     近似 [0.910, 0.000, 0.133]
位置误差:            0.61m (零位)
关节运动后误差:      预计 >>1m（因为关节轴方向也错了）
```

## 根因分析

| 项目 | xArm6 DH 参数 | 当前 MJCF | 影响 |
|------|-------------|-----------|------|
| 关节轴 | 混合（alpha 非零：J2=-π/2, J4=-π/2, J5=π/2, J6=-π/2） | 全部 `axis="0 0 1"` | 运动学完全错误 |
| 连杆长度 | a=[0,0,0.2895,0.0775,0,0] | 纯 X 向偏移: 0.3,0.25,0.12,0.08 | 尺寸不匹配 |
| 连杆偏移 | d=[0.267,0,0,0.3425,0,0.0975] | 无 d 偏移（仅 Z=0.133 base） | 高度不匹配 |
| theta_offset | J2=-π/2 | 无 | 零位姿态不同 |
| 总臂展 | ~0.807m（含 base height 0.267） | 0.91m（水平） | 工作空间不同 |

**核心结论**：当前 MJCF 模型是一个完全不同的通用 6-DOF 臂，不是 xArm6。仿真环境中的 IK 解发给 MuJoCo 后，末端完全不在期望位置。

## 修复方案

### 方案 A：URDF → MJCF 转换（推荐，但工作量大）

- 从 xArm 官方 URDF/Xacro 生成精确 MJCF
- 需要 xacro → URDF 转换（依赖 ROS2 包 `xarm_description`），然后再 URDF → MJCF
- 优势：一次做完，后续不用维护
- 劣势：xacro 依赖 ROS2 环境，不一定能在纯 Python 中转

### 方案 B：按 DH 参数重写 MJCF（更可控，推荐优先尝试）

利用 MuJoCo 的 body `pos` 和 `quat`/`euler` 字段，手工构建符合 DH 参数的运动链：

```xml
<!-- 关节 1：垂直旋转轴，基座高度 d1=0.267 -->
<body name="link1" pos="0 0 0.267">
  <joint name="joint1" type="hinge" axis="0 0 1"/>
  <!-- 关节 2：水平轴（alpha=-π/2），theta_offset=-π/2 -->
  <body name="link2" pos="0 0 0" quat="...">
    <joint name="joint2" type="hinge" axis="0 1 0"/>
    ...
```

关键挑战：
- 每个关节的 `pos` 对应 DH 的 (a, d)，`quat` 对应 alpha 扭角
- MuJoCo body `pos` 是在父坐标系下的偏移
- 需要正确理解 DH 参数定义（标准 DH vs 改进 DH）与 MuJoCo 坐标系的对应关系

### 方案 C：用 Pinocchio 作为 FK 引擎替代 MuJoCo FK（最快，但不完美）

- `MujocoRobotInterface` 中使用 Pinocchio 做 FK 验证 MuJoCo 的 qpos
- MuJoCo 纯粹作为可视化/物理引擎，FK 仍然用准确的手写实现
- 优势：无需修改 MJCF，一天内可完成
- 劣势：MuJoCo 的 FK 仍然不准确，物理仿真（碰撞检测等）会受影响

## 建议

**短期（本周）**：方案 B — 按 DH 参数重写 MJCF。虽然需要仔细处理坐标变换，但逻辑清晰，且完全在掌控内。

**长期**：方案 A — URDF → MJCF 自动化转换，作为基础设施沉淀。

## 工作量估计

- 方案 B：2-4 小时（理解 DH→MJCF 坐标系映射 + 编写 + 验证 FK 误差 <1mm）
- 方案 A：4-8 小时（处理 xacro 依赖、调试 URDF→MJCF 转换精度）
- 验证脚本：30 分钟（对比手写 FK 与 MuJoCo FK 在所有随机配置下的误差分布）

# Stage 2 — 仿真环境调研与实现

> 2026-05-29 调研 → 2026-05-29 实现

## 现状

| 仿真平台 | xArm6 模型 | 备注 |
|----------|-----------|------|
| **MuJoCo Menagerie** | 无 (xArm7 + Lite6 可用) | [Issue #206](https://github.com/google-deepmind/mujoco_menagerie/issues/206) 请求中 |
| **Isaac Gym** | 无现成模型 | 支持 URDF 导入 |
| **xArm 官方** | URDF Xacro | 需 xacro + ROS2 依赖转换 |
| **自建 MJCF** | ✅ 已实现 | 简化 6-DOF 臂模型，用于管道集成测试 |

## 已实现

### MujocoRobotInterface

`stage_2/simulation/mujoco_interface.py` — 完整实现 `RobotInterface` ABC：

```
connect()          → 加载 MJCF 模型，初始化 mjModel/mjData
disconnect()       → 清理 MuJoCo 资源
get_joint_state()  → 从 mjData.qpos/qvel 读取 6 关节 + 力矩
move_joints()      → 设置 actuator ctrl，步进物理仿真（blocking/non-blocking）
stop()             → ctrl = current_qpos
get_gripper_state() / move_gripper() → 夹爪状态（简化实现）
```

### 管道集成

safety_node 新增 `robot_mode="mujoco"`：
- 参数 `mjcf_path` 指定模型文件（默认 `stage_2/simulation/xarm6.xml`）
- `ros2 launch` 文件中设置 `robot_mode:=mujoco` 即可切换

### 端到端验证

```
mock_hand_tracker → hand_mapper → safety(mujoco) → recorder
```

测试结果：
- 8 秒录制：174 步指令 + 174 步状态（~22 Hz，与 mock 模式一致）
- 指令-状态跟踪正常（joint 1 范围：cmd [0.46-0.56], state [0.45-0.56]）
- 36/36 测试通过（无回归）

### 使用方式

```bash
# 在容器内：
python3 /workspace/umi/stage_2/simulation/test_sim_pipeline.py

# 或通过 ROS2 launch（需要先构建并 source）：
# ros2 launch stage_2.simulation teleop_sim.launch.py
```

## 待改进

- [ ] 从 xArm6 URDF 生成精确 MJCF（当前为简化模型，FK 与手写 IK 不匹配）
- [ ] 添加 MuJoCo 可视化（render_offscreen / 图形窗口）
- [ ] 仿真模式下的夹爪物理模拟（当前为占位实现）
- [ ] 重力/碰撞参数调整

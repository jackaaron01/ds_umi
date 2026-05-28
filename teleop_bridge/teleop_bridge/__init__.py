# =============================================================================
# teleop_bridge/ —— Quest3 → ROS2 遥操作桥接
# =============================================================================
# 职责：
#   1. 通过 hand-tracking-sdk 接收 Quest3 手部追踪数据（21 关键点 + 手腕位姿）
#   2. 将手部位姿映射为机器人关节目标或笛卡尔目标
#   3. 发布到 ROS2 topic，供 control/recorder 节点使用
#
# 数据流：
#   Quest3 (Hand Tracking Streamer App)
#     → UDP (WiFi)
#   hand-tracking-sdk-ros2 节点
#     → ROS2 topics: /hand/left/keypoints, /hand/right/keypoints
#   本模块的 hand_mapper 节点
#     → ROS2 topics: /teleop/joint_commands, /teleop/gripper_commands
#   robot_hal/ 的 XArm6Interface 或 MockRobotInterface
#
# 映射方式（后续实现）：
#   - 关节空间映射：手腕位姿 → IK 求解 → 关节角度
#   - 手指映射：手部关键点 → 夹爪开度
# =============================================================================

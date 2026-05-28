# =============================================================================
# kinematics/ —— 自研运动学库
# =============================================================================
# 手写 FK（前向运动学）和 IK（逆运动学），不依赖 ROS。
#
# 目的：
#   1. 学习：搞清楚 FK/IK 的数学原理（DH 参数、雅可比伪逆、数值优化）
#   2. 对比：用 Pinocchio 作为"参考答案"，验证手写实现的正确性
#   3. 迁移：纯 Python 实现可以在任何地方跑（CI、notebook、非 ROS 环境）
#
# 参考资料：
#   - Modern Robotics (Lynch & Park): FK/IK 章节
#   - xArm6 官方 DH 参数
# =============================================================================
from stage_1.kinematics.dh_params import XARM6_DH_PARAMS, XARM6_JOINT_LIMITS, XARM6_VELOCITY_LIMITS
from stage_1.kinematics.fk import forward_kinematics, end_effector_pose, link_transforms
from stage_1.kinematics.jacobian import compute_jacobian
from stage_1.kinematics.ik import solve_ik

__all__ = [
    "XARM6_DH_PARAMS",
    "XARM6_JOINT_LIMITS",
    "XARM6_VELOCITY_LIMITS",
    "forward_kinematics",
    "end_effector_pose",
    "link_transforms",
    "compute_jacobian",
    "solve_ik",
]

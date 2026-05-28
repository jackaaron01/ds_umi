# =============================================================================
# robot_hal/ —— 硬件抽象层（HAL）
# =============================================================================
# 这是整个系统最重要的模块。所有上层代码只依赖 RobotInterface 这个抽象接口，
# 不关心底层是 xArm6、UR5、还是仿真机器人。
#
# 换硬件时：只需写一个新的 XxxInterface 实现类，其余代码零改动。
# 无硬件时：MockRobotInterface 让整个遥操作-录制流水线在没有真机器人的情况下跑通。
#
# 设计原则：
#   - 纯 Python，不依赖 ROS2（可以在任何环境做单元测试）
#   - 接口方法以关节空间操作为主（遥操作的自然抽象层级）
#   - 笛卡尔空间操作作为扩展接口（optional）
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class JointState:
    """关节状态数据类"""
    position: np.ndarray          # 关节角度 (rad)
    velocity: np.ndarray          # 关节角速度 (rad/s)
    effort: np.ndarray            # 关节力矩 (Nm)
    name: List[str]               # 关节名称（与 URDF 中的 joint name 对应）


@dataclass
class GripperState:
    """夹爪状态数据类"""
    position: float               # 夹爪开度 [0, 1]，0=闭合 1=完全张开
    effort: float                 # 夹持力 (N) 或归一化值


class RobotInterface(ABC):
    """
    机器人抽象接口 —— 所有具体机器人实现的基类。

    上层代码只 import RobotInterface，不知道下面是什么机器人。
    这种设计叫"依赖倒置原则"（Dependency Inversion Principle）：
    高层模块不依赖低层模块，两者都依赖抽象。

    使用示例：
        robot = XArm6Interface(ip="192.168.1.100")
        robot.connect()
        state = robot.get_joint_state()
        robot.move_joints(target_positions)
        robot.disconnect()
    """

    @abstractmethod
    def connect(self) -> bool:
        """建立与机器人的连接，返回 True 表示成功"""
        ...

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接，安全关闭"""
        ...

    @abstractmethod
    def get_joint_state(self) -> JointState:
        """读取当前关节状态"""
        ...

    @abstractmethod
    def move_joints(self,
                    positions: np.ndarray,
                    velocity: float = 0.5,
                    blocking: bool = True) -> bool:
        """
        关节空间运动指令。
        blocking=True 时等运动完成再返回（适合录制数据的场景）。
        blocking=False 时立即返回（适合实时遥操作的场景）。
        """
        ...

    @abstractmethod
    def stop(self) -> bool:
        """紧急停止，最高优先级"""
        ...

    @abstractmethod
    def get_gripper_state(self) -> GripperState:
        """读取夹爪状态"""
        ...

    @abstractmethod
    def move_gripper(self, position: float, blocking: bool = True) -> bool:
        """控制夹爪开度 [0, 1]"""
        ...

    # ---- 可选扩展接口 ----
    # 这些方法有默认实现（抛 NotImplementedError），子类可选实现

    def move_cartesian(self,
                       pose: np.ndarray,      # [x, y, z, roll, pitch, yaw]
                       velocity: float = 0.1,
                       blocking: bool = True) -> bool:
        """笛卡尔空间运动（可选）"""
        raise NotImplementedError("Cartesian control not supported by this robot")

    def get_cartesian_pose(self) -> np.ndarray:
        """获取末端笛卡尔位姿（可选）"""
        raise NotImplementedError("Cartesian pose not supported by this robot")
from stage_1.robot_hal.mock_robot import MockRobotInterface
from stage_1.robot_hal.xarm6_interface import XArm6Interface

__all__ = ["MockRobotInterface", "XArm6Interface"]

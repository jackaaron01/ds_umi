import numpy as np
import pytest
from stage_1.kinematics.dh_params import XARM6_DH_PARAMS, XARM6_JOINT_LIMITS


@pytest.fixture
def dh_params():
    return XARM6_DH_PARAMS


@pytest.fixture
def joint_limits():
    return XARM6_JOINT_LIMITS


@pytest.fixture
def random_q():
    """Random valid joint configuration (seeded for reproducibility)."""
    rng = np.random.RandomState(42)
    # Stay away from joint limits
    q = rng.uniform(-1.5, 1.5, 6)
    # Clamp to limits
    return np.clip(q, XARM6_JOINT_LIMITS[:, 0], XARM6_JOINT_LIMITS[:, 1])


@pytest.fixture
def sample_configs():
    """Multiple diverse configurations to test with."""
    configs = [
        np.zeros(6),                          # home
        np.array([0.0, -np.pi/4, 0.0, np.pi/4, 0.0, 0.0]),
        np.array([np.pi/6, -np.pi/3, np.pi/4, -np.pi/6, np.pi/3, -np.pi/4]),
        np.array([1.0, -1.0, 0.5, -0.5, 1.0, -1.0]),
        np.array([-1.5, -0.5, 1.5, 0.8, -1.2, 0.3]),
    ]
    # Clamp all to limits
    clamped = []
    for q in configs:
        clamped.append(np.clip(q, XARM6_JOINT_LIMITS[:, 0], XARM6_JOINT_LIMITS[:, 1]))
    return clamped

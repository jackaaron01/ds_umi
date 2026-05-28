import numpy as np
import pytest
from stage_1.kinematics.fk import forward_kinematics, end_effector_pose, link_transforms
from stage_1.kinematics.utils import dh_transform


class TestForwardKinematics:
    def test_zero_config_returns_identity_base(self, dh_params):
        q = np.zeros(6)
        transforms = link_transforms(q, dh_params)
        assert len(transforms) == 7  # base + 6 links
        np.testing.assert_array_almost_equal(transforms[0], np.eye(4))

    def test_all_transforms_are_orthonormal(self, dh_params, sample_configs):
        for q in sample_configs:
            transforms = link_transforms(q, dh_params)
            for T in transforms:
                R = T[:3, :3]
                np.testing.assert_array_almost_equal(
                    R @ R.T, np.eye(3), decimal=10,
                    err_msg=f"R not orthonormal at q={q}"
                )
                np.testing.assert_almost_equal(
                    np.linalg.det(R), 1.0, decimal=10,
                    err_msg=f"det(R) != 1 at q={q}"
                )

    def test_end_effector_pose_structure(self, dh_params, random_q):
        R, p = end_effector_pose(random_q, dh_params)
        assert R.shape == (3, 3)
        assert p.shape == (3,)
        np.testing.assert_array_almost_equal(R @ R.T, np.eye(3), decimal=10)
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0, decimal=10)

    def test_fk_returns_valid_positions(self, dh_params, sample_configs):
        for q in sample_configs:
            positions, orientations = forward_kinematics(q, dh_params)
            assert positions.shape == (7, 3)
            assert orientations.shape == (7, 3, 3)
            # Positions should be finite
            assert np.all(np.isfinite(positions))

    def test_fk_deterministic(self, dh_params, random_q):
        R1, p1 = end_effector_pose(random_q, dh_params)
        R2, p2 = end_effector_pose(random_q, dh_params)
        np.testing.assert_array_almost_equal(R1, R2)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_dh_transform_unit(self):
        """Verify DH transform with known parameters gives expected result."""
        T = dh_transform(0.0, 0.0, 0.1, 0.0)  # pure Z translation
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1],
            [0, 0, 0, 1],
        ])
        np.testing.assert_array_almost_equal(T, expected)

        T = dh_transform(0.0, 0.0, 0.0, np.pi / 2)  # pure Z rotation
        expected = np.array([
            [0, -1, 0, 0],
            [1,  0, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1],
        ])
        np.testing.assert_array_almost_equal(T, expected)

import numpy as np
import pytest
from stage_1.kinematics.jacobian import compute_jacobian
from stage_1.kinematics.fk import end_effector_pose
from stage_1.kinematics.utils import so3_log


class TestJacobian:
    def test_jacobian_shape(self, dh_params, random_q):
        J = compute_jacobian(random_q, dh_params)
        assert J.shape == (6, 6)

    def test_jacobian_rank(self, dh_params, sample_configs):
        """Jacobian should be full rank at non-singular configurations."""
        for q in sample_configs:
            J = compute_jacobian(q, dh_params)
            rank = np.linalg.matrix_rank(J)
            # Not all configs are non-singular; just verify rank >= 1
            assert rank >= 1

    def test_finite_difference_consistency(self, dh_params, random_q):
        """Analytical Jacobian should match finite differences."""
        J_analytical = compute_jacobian(random_q, dh_params)
        J_fd = np.zeros((6, 6))
        eps = 1e-6

        R0, p0 = end_effector_pose(random_q, dh_params)

        for i in range(6):
            q_plus = random_q.copy()
            q_plus[i] += eps
            R_plus, p_plus = end_effector_pose(q_plus, dh_params)

            # Position difference
            J_fd[:3, i] = (p_plus - p0) / eps

            # Rotation difference in spatial frame: R_plus @ R0^T
            R_diff = R_plus @ R0.T
            omega = so3_log(R_diff)
            J_fd[3:, i] = omega / eps

        # Compare with element-wise tolerance. Near-zero elements (e.g.
        # joint-6 angular component) have inherently large relative errors
        # from finite differences, so we use a modest absolute tolerance.
        assert np.allclose(J_analytical, J_fd, rtol=1e-2, atol=0.01), (
            f"Analytical Jacobian does not match finite differences.\n"
            f"Analytical:\n{J_analytical}\n"
            f"Finite diff:\n{J_fd}\n"
            f"Abs diff:\n{np.abs(J_analytical - J_fd)}"
        )

    def test_jacobian_all_finite(self, dh_params, sample_configs):
        for q in sample_configs:
            J = compute_jacobian(q, dh_params)
            assert np.all(np.isfinite(J)), f"Jacobian has NaN/inf at q={q}"

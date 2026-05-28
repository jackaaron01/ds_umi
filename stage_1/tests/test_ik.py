import numpy as np
import pytest
from stage_1.kinematics.ik import solve_ik
from stage_1.kinematics.fk import end_effector_pose
from stage_1.kinematics.utils import pose_error


class TestInverseKinematics:
    def test_self_consistency_from_home(self, dh_params, joint_limits):
        """IK should recover the home config when targeting the home pose."""
        q0 = np.zeros(6)
        R, p = end_effector_pose(q0, dh_params)
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = p

        q_sol, success, iters, err = solve_ik(
            T_target, q_init=q0, dh_params=dh_params, joint_limits=joint_limits
        )
        # Should converge trivially (already at target)
        assert success, f"Trivial IK should succeed, got error={err:.6f}"
        assert iters <= 3, f"Trivial IK should converge in <= 3 iterations, got {iters}"

    def test_small_perturbation_recovery(self, dh_params, joint_limits):
        """IK should recover from a small perturbation at a non-singular config."""
        # Use a well-conditioned configuration (not near singularity)
        q_target = np.array([0.5, -0.8, 1.2, -0.6, 0.3, -0.4])
        R, p = end_effector_pose(q_target, dh_params)
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = p

        q_init = q_target + 0.05  # small perturbation
        q_sol, success, iters, err = solve_ik(
            T_target, q_init=q_init, dh_params=dh_params, joint_limits=joint_limits
        )
        assert success, f"IK should recover from small perturbation, error={err:.6f}"

    def test_returns_best_effort_on_failure(self, dh_params, joint_limits):
        """Even when IK can't converge, it should return a valid solution."""
        # Target pose far outside workspace
        T_far = np.eye(4)
        T_far[:3, 3] = [5.0, 0.0, 0.0]  # 5 meters away
        q_init = np.zeros(6)

        q_sol, success, iters, err = solve_ik(
            T_far, q_init=q_init, dh_params=dh_params, joint_limits=joint_limits,
            max_iterations=50,
        )
        # Probably won't converge, but solution should be within joint limits
        for i in range(6):
            assert joint_limits[i, 0] - 1e-6 <= q_sol[i] <= joint_limits[i, 1] + 1e-6

    def test_solution_within_tolerance_on_success(self, dh_params, joint_limits):
        """When IK reports success, verify the solution is actually within tolerance."""
        q_target = np.array([0.5, -0.8, 1.2, -0.6, 0.3, -0.4])
        R, p = end_effector_pose(q_target, dh_params)
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = p

        q_init = q_target + 0.1
        q_sol, success, _, _ = solve_ik(
            T_target, q_init=q_init, dh_params=dh_params, joint_limits=joint_limits
        )
        if success:
            R_sol, p_sol = end_effector_pose(q_sol, dh_params)
            T_sol = np.eye(4)
            T_sol[:3, :3] = R_sol
            T_sol[:3, 3] = p_sol
            e = pose_error(T_sol, T_target)
            assert np.linalg.norm(e[:3]) < 1e-2  # position within 1 cm
            assert np.linalg.norm(e[3:]) < 1e-2  # rotation within 0.01 rad

    def test_joint_limits_respected(self, dh_params, joint_limits):
        """IK should never return a solution outside joint limits."""
        q_target = np.array([0.5, -0.8, 1.2, -0.6, 0.3, -0.4])
        R, p = end_effector_pose(q_target, dh_params)
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = p

        # Start near a limit
        q_init = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
        q_sol, _, _, _ = solve_ik(
            T_target, q_init=q_init, dh_params=dh_params, joint_limits=joint_limits
        )
        for i in range(6):
            assert joint_limits[i, 0] - 1e-6 <= q_sol[i] <= joint_limits[i, 1] + 1e-6, \
                f"Joint {i} out of bounds: {q_sol[i]}"

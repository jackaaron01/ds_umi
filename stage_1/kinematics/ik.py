import numpy as np
from stage_1.kinematics.dh_params import XARM6_DH_PARAMS, XARM6_JOINT_LIMITS
from stage_1.kinematics.fk import end_effector_pose
from stage_1.kinematics.jacobian import compute_jacobian
from stage_1.kinematics.utils import pose_error


def solve_ik(
    target_pose: np.ndarray,
    q_init: np.ndarray = None,
    dh_params: np.ndarray = XARM6_DH_PARAMS,
    joint_limits: np.ndarray = XARM6_JOINT_LIMITS,
    max_iterations: int = 200,
    pos_tolerance: float = 1e-3,
    rot_tolerance: float = 1e-3,
    lambda_min: float = 0.01,
    lambda_max: float = 1.0,
) -> tuple:
    """Solve inverse kinematics using damped least squares (Levenberg-Marquardt).

    Args:
        target_pose: 4x4 homogeneous transform of desired end-effector pose
        q_init: initial joint configuration [6,], defaults to zeros if None
        dh_params: DH parameter table
        joint_limits: (6, 2) array of [min, max] per joint
        max_iterations: maximum number of solver iterations
        pos_tolerance: position error convergence threshold (m)
        rot_tolerance: rotation error convergence threshold (rad)
        lambda_min: minimum damping factor
        lambda_max: maximum damping factor

    Returns:
        (q_solution, success, iterations, final_error)
        q_solution: best joint configuration found [6,]
        success: True if converged within tolerance
        iterations: number of iterations used
        final_error: norm of final pose error (6-vector)
    """
    if q_init is None:
        q_init = np.zeros(6)

    q = q_init.copy().astype(np.float64)
    best_q = q.copy()
    best_error = np.inf
    lam = 0.1  # initial damping

    for iteration in range(max_iterations):
        R_cur, p_cur = end_effector_pose(q, dh_params)
        T_current = np.eye(4)
        T_current[:3, :3] = R_cur
        T_current[:3, 3] = p_cur

        e = pose_error(T_current, target_pose)
        error_norm = np.linalg.norm(e)

        if error_norm < best_error:
            best_error = error_norm
            best_q = q.copy()

        pos_err = np.linalg.norm(e[:3])
        rot_err = np.linalg.norm(e[3:])
        if pos_err < pos_tolerance and rot_err < rot_tolerance:
            return q.copy(), True, iteration + 1, error_norm

        J = compute_jacobian(q, dh_params)

        damping = lam * lam * np.eye(6)
        JTJ = J.T @ J
        try:
            dq = np.linalg.solve(JTJ + damping, J.T @ e)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(JTJ + damping, J.T @ e, rcond=None)[0]

        # Line search: try decreasing step sizes
        accepted = False
        for alpha in [1.0, 0.5, 0.25]:
            q_candidate = q + alpha * dq
            q_candidate = np.clip(q_candidate, joint_limits[:, 0], joint_limits[:, 1])

            R_cand, p_cand = end_effector_pose(q_candidate, dh_params)
            T_candidate = np.eye(4)
            T_candidate[:3, :3] = R_cand
            T_candidate[:3, 3] = p_cand

            e_candidate = pose_error(T_candidate, target_pose)
            candidate_error = np.linalg.norm(e_candidate)
            if candidate_error < error_norm:
                q = q_candidate
                lam = max(lam * 0.5, lambda_min)
                accepted = True
                break

        if accepted:
            continue

        # Increase damping when step failed
        lam = min(lam * 2.0, lambda_max)
        # Use small step with higher damping in next iteration
        q = np.clip(q + 0.25 * dq, joint_limits[:, 0], joint_limits[:, 1])

    return best_q, best_error < 1e-2, max_iterations, best_error

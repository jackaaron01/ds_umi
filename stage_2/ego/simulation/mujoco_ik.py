#!/usr/bin/env python3
"""
MuJoCo-based Inverse Kinematics for xArm6 mesh model.

Uses MuJoCo's mj_jac to compute the end-effector Jacobian directly from
the loaded MJCF model, then iteratively solves for joint angles using
damped pseudoinverse.

This replaces the DH-parameter-based IK when using the mesh (URDF) model,
since the mesh model has different kinematics than the DH model.
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class MujocoIK:
    """Numerical IK solver using MuJoCo Jacobian for the xArm6 mesh model."""

    def __init__(self, model_path: str, site_name: str = "ee"):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")
        # Get the body that contains this site (mj_jac needs body id, not site id!)
        self._body_id = self._model.site_bodyid[self._site_id]
        self._nv = self._model.nv  # number of DoFs (6 for xArm6)

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        q_init: np.ndarray = None,
        q_nominal: np.ndarray = None,
        max_iter: int = 50,
        tolerance: float = 1e-3,
        damping: float = 0.05,
    ) -> np.ndarray:
        """
        Solve IK for target end-effector pose.

        Uses damped pseudoinverse with null-space control that pulls joints
        toward a nominal (home) configuration. This prevents the IK from
        oscillating between different solutions for the same end-effector pose.

        Args:
            target_pos: [3] target position in world frame
            target_quat: [4] target orientation quaternion (x, y, z, w)
            q_init: initial joint config. If None, uses zeros.
            q_nominal: preferred joint config for null-space control.
                       If None, uses a standard home pose.
            max_iter: maximum iterations (reduced for incremental tracking)
            tolerance: convergence tolerance
            damping: damping factor (higher = more stable)

        Returns:
            q: [nv] joint angles
        """
        nv = min(self._nv, 6)
        if q_init is not None:
            self._data.qpos[:nv] = q_init[:nv].copy()
        else:
            self._data.qpos[:nv] = 0.0

        if q_nominal is None:
            q_nominal = np.array([0.0, -0.3, 0.0, 1.2, 0.0, 0.0])
        else:
            q_nominal = np.asarray(q_nominal, dtype=np.float64)[:nv]

        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_quat_xyzw = None
        if target_quat is not None:
            target_quat_xyzw = np.asarray(target_quat, dtype=np.float64)
            if len(target_quat_xyzw) != 4:
                raise ValueError(f"Quaternion must have 4 elements")

        best_q = self._data.qpos[:nv].copy()
        best_error = float("inf")
        nullspace_gain = 0.05  # moderate pull toward nominal

        for _ in range(max_iter):
            mujoco.mj_forward(self._model, self._data)

            site_pos = self._data.site_xpos[self._site_id].copy()
            site_mat = self._data.site_xmat[self._site_id].reshape(3, 3)

            # Position error
            pos_err = target_pos - site_pos

            # Orientation error (if applicable)
            if target_quat_xyzw is not None:
                target_rot = Rotation.from_quat(target_quat_xyzw)
                site_rot = Rotation.from_matrix(site_mat)
                rot_err = target_rot * site_rot.inv()
                rot_vec = rot_err.as_rotvec()
                err = np.concatenate([pos_err, rot_vec])
            else:
                err = pos_err

            error_norm = np.linalg.norm(err)
            if error_norm < best_error:
                best_error = error_norm
                best_q = self._data.qpos[:nv].copy()

            if error_norm < tolerance:
                return best_q

            # Compute Jacobian (position + rotation)
            jac_pos = np.zeros((3, self._nv))
            jac_rot = np.zeros((3, self._nv))
            mujoco.mj_jac(self._model, self._data, jac_pos, jac_rot,
                          self._data.site_xpos[self._site_id], self._body_id)

            if target_quat_xyzw is not None:
                jac = np.vstack([jac_pos[:, :nv], jac_rot[:, :nv]])
                m = 6
            else:
                jac = jac_pos[:, :nv]
                m = 3

            # Damped pseudoinverse: J⁺ = J^T (J J^T + λ² I)^{-1}
            jjt = jac @ jac.T
            damp_mat = damping ** 2 * np.eye(m)
            try:
                j_inv = jac.T @ np.linalg.solve(jjt + damp_mat, np.eye(m))
            except np.linalg.LinAlgError:
                j_inv = damping * jac.T  # fallback to transpose

            # Task-space correction
            delta_q_task = j_inv @ err

            # Null-space correction: pull toward nominal pose
            null_proj = np.eye(nv) - j_inv @ jac
            delta_q_null = nullspace_gain * null_proj @ (q_nominal - self._data.qpos[:nv])

            delta_q = delta_q_task + delta_q_null

            # Strict step size limit per iteration
            max_delta = 0.08  # rad per iteration (small incremental steps)
            delta_norm = np.linalg.norm(delta_q)
            if delta_norm > max_delta:
                delta_q *= max_delta / delta_norm

            self._data.qpos[:nv] += delta_q

            # Clamp to joint limits
            for j in range(min(nv, self._model.njnt)):
                jnt_id = self._model.jnt_qposadr[j]
                if self._model.jnt_limited[j]:
                    lo = self._model.jnt_range[j][0]
                    hi = self._model.jnt_range[j][1]
                    self._data.qpos[jnt_id] = np.clip(self._data.qpos[jnt_id], lo, hi)

        return best_q

    def fk(self, q: np.ndarray) -> tuple:
        """
        Forward kinematics: compute end-effector pose for given joint angles.

        Returns:
            (position [3], rotation_matrix [3,3])
        """
        nv = min(self._nv, 6)
        self._data.qpos[:nv] = q[:nv]
        mujoco.mj_forward(self._model, self._data)
        pos = self._data.site_xpos[self._site_id].copy()
        rot = self._data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pos, rot

    @property
    def n_joints(self) -> int:
        return min(self._nv, 6)

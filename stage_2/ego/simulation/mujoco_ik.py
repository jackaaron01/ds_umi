#!/usr/bin/env python3
"""
MuJoCo-based 6-DOF Inverse Kinematics for xArm6.

Uses MuJoCo's mj_jac with damped pseudoinverse, enhanced with features
from the orin_VR Pinocchio/CasADi/IPOPT solver:

  - Weighted SE(3) error (position vs orientation trade-off)
  - Per-joint regularization toward nominal pose
  - Temporal smoothness (penalize deviation from previous solution)
  - Trust region (per-step joint delta limit)
  - Wrist branch stabilization (joint4/5/6 flip ambiguity)
  - Joint limit clamping
"""

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


class MujocoIK:
    """Numerical 6-DOF IK solver using MuJoCo Jacobian + damped pseudoinverse."""

    def __init__(self, model_path: str, site_name: str = "ee", n_joints: int = 6):
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._site_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in model")
        self._body_id = self._model.site_bodyid[self._site_id]
        self._nv_full = self._model.nv
        self._nv = min(n_joints, self._nv_full)

        # Previous solution for temporal smoothness
        self.q_last = np.zeros(self._nv)

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray = None,
        q_init: np.ndarray = None,
        q_nominal: np.ndarray = None,
        max_iter: int = 50,
        tolerance: float = 1e-3,
        damping: float = 0.05,
        pos_weight: float = 45.0,
        rot_weight: float = 5.0,
        reg_weights: np.ndarray = None,
        smooth_weight: float = 0.05,
        trust_region: float = 0.3,
    ) -> np.ndarray:
        """
        Solve IK for target end-effector pose.

        Args:
            target_pos:  [3] target position in world frame (m)
            target_quat: [4] target orientation quaternion (x,y,z,w)
            q_init:      initial joint config for this solve
            q_nominal:   preferred joint config (null-space target)
            max_iter:    max iterations
            tolerance:   convergence tolerance (weighted error norm)
            damping:     numerical damping for pseudoinverse
            pos_weight:  position error weight (higher = prioritize position)
            rot_weight:  orientation error weight
            reg_weights: per-joint regularization [6] (higher = penalize more)
            smooth_weight: temporal smoothness (higher = stay closer to q_last)
            trust_region: max per-step joint change (rad)

        Returns:
            q: [nv] joint angles for arm joints
        """
        nv = self._nv

        # Init q
        if q_init is not None:
            self._data.qpos[:nv] = np.asarray(q_init[:nv], dtype=np.float64).copy()
            q_start = self._data.qpos[:nv].copy()
        else:
            self._data.qpos[:nv] = 0.0
            q_start = np.zeros(nv)

        # Nominal pose
        if q_nominal is None:
            q_nominal = np.array([0.0, -0.6109, -0.6981, 0.0, 1.3788, 0.0])
        q_nominal = np.asarray(q_nominal, dtype=np.float64)[:nv]

        # Regularization weights (default: penalize j1/j4 more, like orin_VR)
        if reg_weights is None:
            reg_weights = np.array([5.0, 1.0, 1.0, 5.0, 1.0, 1.0])
        reg_weights = np.asarray(reg_weights, dtype=np.float64)[:nv]

        # Target
        target_pos = np.asarray(target_pos, dtype=np.float64)
        use_orientation = target_quat is not None
        if use_orientation:
            target_quat_xyzw = np.asarray(target_quat, dtype=np.float64)
            if len(target_quat_xyzw) != 4:
                raise ValueError("Quaternion must have 4 elements")

        best_q = self._data.qpos[:nv].copy()
        best_cost = float("inf")

        for _ in range(max_iter):
            mujoco.mj_forward(self._model, self._data)

            # ── Build weighted error ──
            site_pos = self._data.site_xpos[self._site_id]
            site_mat = self._data.site_xmat[self._site_id].reshape(3, 3)

            pos_err = target_pos - site_pos

            if use_orientation:
                site_rot = Rotation.from_matrix(site_mat)
                target_rot = Rotation.from_quat(target_quat_xyzw)
                # Rotation error: log(R_target * R_site^T) as rotvec
                rot_err = target_rot * site_rot.inv()
                rot_vec = rot_err.as_rotvec()
                # Weighted error
                err = np.concatenate([
                    pos_weight * pos_err, rot_weight * rot_vec])
            else:
                err = pos_err

            # ── Cost (for best-solution tracking) ──
            task_cost = 0.5 * float(np.linalg.norm(err) ** 2)
            reg_cost = 0.5 * float(
                np.sum(reg_weights * (self._data.qpos[:nv] - q_nominal) ** 2))
            smooth_cost = 0.5 * smooth_weight * float(
                np.sum((self._data.qpos[:nv] - self.q_last[:nv]) ** 2))
            total_cost = task_cost + reg_cost + smooth_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_q = self._data.qpos[:nv].copy()

            # Check convergence
            if np.max(np.abs(pos_err)) < tolerance and (
                not use_orientation or np.linalg.norm(rot_vec) < tolerance):
                break

            # ── Jacobian ──
            jac_pos = np.zeros((3, self._nv_full))
            jac_rot = np.zeros((3, self._nv_full))
            mujoco.mj_jac(self._model, self._data, jac_pos, jac_rot,
                          self._data.site_xpos[self._site_id], self._body_id)

            if use_orientation:
                # Weighted Jacobian
                J = np.vstack([
                    pos_weight * jac_pos[:, :nv],
                    rot_weight * jac_rot[:, :nv],
                ])
                m = 6
            else:
                J = jac_pos[:, :nv]
                m = 3

            # ── Damped pseudoinverse ──
            JJT = J @ J.T
            damp_mat = damping ** 2 * np.eye(m)
            try:
                J_inv = J.T @ np.linalg.solve(JJT + damp_mat, np.eye(m))
            except np.linalg.LinAlgError:
                J_inv = damping * J.T

            # ── Task-space correction ──
            dq_task = J_inv @ err

            # ── Null-space: pull toward q_nominal ──
            null_proj = np.eye(nv) - J_inv @ J
            # Regularization gradient in joint space
            grad_reg = -reg_weights * (self._data.qpos[:nv] - q_nominal)
            # Smoothness gradient
            grad_smooth = -smooth_weight * (self._data.qpos[:nv] - self.q_last[:nv])
            dq_null = 0.02 * null_proj @ (grad_reg + grad_smooth)

            dq = dq_task + dq_null

            # ── Trust region (per-step limit) ──
            max_step = trust_region
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n

            # ── Joint limit clamping ──
            self._data.qpos[:nv] += dq
            for j in range(min(nv, self._model.njnt)):
                if self._model.jnt_limited[j]:
                    jid = self._model.jnt_qposadr[j]
                    lo, hi = self._model.jnt_range[j]
                    self._data.qpos[jid] = np.clip(
                        self._data.qpos[jid], lo + 0.01, hi - 0.01)

        # ── Post-solve: wrist branch + base yaw stabilization ──
        best_q = self._stabilize_wrist(best_q, q_start)
        best_q = self._stabilize_base_yaw(best_q, q_start)

        self.q_last = best_q.copy()
        return best_q

    def _stabilize_wrist(self, q: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        """Resolve wrist flip ambiguity: (j4+π, -j5, j6+π) gives same pose."""
        if self._nv < 6:
            return q
        candidates = [q]
        # Flipped branch
        q_flip = q.copy()
        q_flip[3] = np.arctan2(np.sin(q[3] + np.pi), np.cos(q[3] + np.pi))
        q_flip[4] = -q[4]
        q_flip[5] = np.arctan2(np.sin(q[5] + np.pi), np.cos(q[5] + np.pi))
        # Clamp to limits
        for qc in [q_flip]:
            for j in range(min(6, self._model.njnt)):
                if self._model.jnt_limited[j]:
                    jid = self._model.jnt_qposadr[j]
                    lo, hi = self._model.jnt_range[j]
                    qc[j] = np.clip(qc[j], lo + 0.01, hi - 0.01)
        candidates.append(q_flip)
        # Pick closest to previous
        best = min(candidates, key=lambda qc: np.sum((qc - q_prev) ** 2))
        return best

    def _stabilize_base_yaw(self, q: np.ndarray, q_prev: np.ndarray) -> np.ndarray:
        """Resolve 2π wrapping for joint1."""
        if self._nv < 1:
            return q
        candidates = [q]
        for offset in [2 * np.pi, -2 * np.pi]:
            q2 = q.copy()
            q2[0] += offset
            candidates.append(q2)
        best = min(candidates, key=lambda qc: np.sum((qc - q_prev) ** 2))
        return best

    def set_q_last(self, q: np.ndarray):
        """Set previous solution (e.g., on grip engage)."""
        self.q_last = np.asarray(q[:self._nv], dtype=np.float64).copy()

    def fk(self, q: np.ndarray) -> tuple:
        nv = self._nv
        self._data.qpos[:nv] = np.asarray(q[:nv], dtype=np.float64)
        mujoco.mj_forward(self._model, self._data)
        pos = self._data.site_xpos[self._site_id].copy()
        rot = self._data.site_xmat[self._site_id].reshape(3, 3).copy()
        return pos, rot

    @property
    def n_joints(self) -> int:
        return self._nv

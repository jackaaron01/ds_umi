#!/usr/bin/env python3
"""
EGO Teleop Simulator — Single self-contained MuJoCo simulation.

Receives hand tracking data via UDP and drives the xArm6 mesh model.
No ROS2 dependency — pure Python + MuJoCo + socket.

Usage (in Docker):
    python3 mujoco_ego_sim.py [--port 9999]

Host runs:  python3 stage_2/ego/mediapipe_ego.py --udp
"""

import sys, os, time, socket, json, struct, argparse, threading
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
from mujoco.viewer import launch_passive

# ── Model ──────────────────────────────────────────────────────────────
MODEL = os.path.join(os.path.dirname(__file__), "xarm_mesh.xml")
N_JOINTS = 6
HOME = np.array([0.0, -0.3, 0.0, 1.2, 0.0, 0.0])

# Reachable workspace bounds (meters from base)
WS_X = (0.0, 0.35)    # forward
WS_Y = (-0.25, 0.25)  # left/right
WS_Z = (0.10, 0.45)   # up


class EgoSimulator:
    """MuJoCo simulation of xArm6 driven by hand-tracking UDP data."""

    def __init__(self, model_path=MODEL):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.body_id = self.model.site_bodyid[self.site_id]

        # Init to home pose
        self.data.qpos[:N_JOINTS] = HOME
        self.data.ctrl[:N_JOINTS] = HOME
        mujoco.mj_forward(self.model, self.data)

        # Latest hand target (thread-safe)
        self._lock = threading.Lock()
        self._target_pos = None  # [x, y, z] in robot workspace
        self._running = True

        # IK state
        self._q = HOME.copy()
        self._home = HOME.copy()

        # ── Camera control ──────────────────────────────────────────────
        self._cam_ego_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
        self._cam_fixed_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed")
        self._cam_mode = "ego"  # default to first-person view
        self._cam_lock = threading.Lock()
        self._viewer = None  # set in run()

        if self._cam_ego_id < 0:
            print("[sim] WARNING: 'ego' camera not found in model")
        if self._cam_fixed_id < 0:
            print("[sim] WARNING: 'fixed' camera not found in model")

    # ── UDP server ──────────────────────────────────────────────────
    def start_udp(self, port=9999):
        """Listen for MediaPipe hand data on UDP (non-blocking)."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.001)
        self._udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        self._udp_thread.start()
        print(f"[sim] UDP listening on port {port}")

    def _udp_loop(self):
        while self._running:
            try:
                data, _ = self._sock.recvfrom(65536)
                msg = json.loads(data.decode("utf-8"))
                wrist = msg.get("wrist", [0, 0, 0, 0, 0, 0, 1])
                # wrist is [x, y, z, qx, qy, qz, qw] in robot workspace
                with self._lock:
                    self._target_pos = np.array(wrist[:3], dtype=np.float64)
            except socket.timeout:
                pass
            except Exception:
                pass

    # ── IK solver ──────────────────────────────────────────────────
    def solve_ik(self, target_pos, q_init, max_iter=40):
        """Position-only IK with null-space pull toward home."""
        nv = min(N_JOINTS, self.model.nv)
        self.data.qpos[:nv] = q_init[:nv].copy()
        target = np.asarray(target_pos, dtype=np.float64)
        best_q = q_init[:nv].copy()
        best_err = float("inf")
        damping = 0.1
        null_gain = 0.03
        max_step = 0.08

        for _ in range(max_iter):
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.site_id]
            err = target - pos
            err_norm = float(np.linalg.norm(err))

            if err_norm < best_err:
                best_err = err_norm
                best_q = self.data.qpos[:nv].copy()
            if err_norm < 0.001:
                return best_q

            # Jacobian
            jac = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jac, jac_rot,
                          self.data.site_xpos[self.site_id], self.body_id)
            J = jac[:, :nv]  # 3×6

            # Damped pseudoinverse
            JJT = J @ J.T
            try:
                J_inv = J.T @ np.linalg.solve(JJT + damping**2 * np.eye(3), np.eye(3))
            except np.linalg.LinAlgError:
                J_inv = damping * J.T

            # Task space correction
            dq_task = J_inv @ err

            # Null space: pull toward home
            null_proj = np.eye(nv) - J_inv @ J
            dq_null = null_gain * null_proj @ (self._home[:nv] - self.data.qpos[:nv])

            dq = dq_task + dq_null

            # Clamp step
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n

            self.data.qpos[:nv] += dq

            # Joint limits
            for j in range(min(nv, self.model.njnt)):
                if self.model.jnt_limited[j]:
                    jid = self.model.jnt_qposadr[j]
                    lo, hi = self.model.jnt_range[j]
                    self.data.qpos[jid] = np.clip(self.data.qpos[jid], lo, hi)

        return best_q

    # ── Camera switching ───────────────────────────────────────────────
    def _on_viewer_key(self, keycode: int):
        """Handle keyboard shortcuts in MuJoCo viewer (GLFW thread).

        Keys:
            1 → Ego camera (EE-mounted, first-person)
            2 → Fixed camera (world-frame overview)
            3 / Space → Free camera (mouse-controlled fly)
        """
        if keycode == 49:       # '1'
            new_mode = "ego"
        elif keycode == 50:     # '2'
            new_mode = "fixed"
        elif keycode in (51, 32):  # '3' or Space
            new_mode = "free"
        else:
            return

        with self._cam_lock:
            if new_mode == self._cam_mode:
                return
            self._cam_mode = new_mode
            name = {"ego": "EGO (EE follow)", "fixed": "FIXED (overview)",
                    "free": "FREE (mouse fly)"}[new_mode]
            print(f"[sim] Camera: {name}")

        # Apply immediately (we're inside viewer's render lock)
        if self._viewer is not None:
            self._apply_camera_mode()

    def _apply_camera_mode(self):
        """Set viewer camera based on current mode. Call from main or GLFW thread."""
        v = self._viewer
        if v is None:
            return
        with self._cam_lock:
            mode = self._cam_mode
        if mode == "ego" and self._cam_ego_id >= 0:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = self._cam_ego_id
        elif mode == "fixed" and self._cam_fixed_id >= 0:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            v.cam.fixedcamid = self._cam_fixed_id
        else:  # free
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # ── Main loop ──────────────────────────────────────────────────
    def run(self):
        """Open viewer and run simulation loop."""
        print(f"[sim] Opening xArm6 viewer...")
        with launch_passive(
            self.model, self.data,
            key_callback=self._on_viewer_key,
        ) as viewer:
            self._viewer = viewer

            # Default to ego (first-person) camera
            self._apply_camera_mode()
            print(f"[sim] Camera: EGO (EE follow) — "
                  f"press 1=Ego  2=Fixed  3/Space=Free")
            print(f"[sim] Running — move your hand in front of camera")
            step_count = 0
            while self._running and viewer.is_running():
                # Get latest target
                with self._lock:
                    target = self._target_pos

                if target is not None:
                    # Solve IK from current simulated joint state
                    q_current = self.data.qpos[:N_JOINTS].copy()
                    q_sol = self.solve_ik(target, q_init=q_current)
                    self._q = q_sol
                    # Set position servo targets
                    self.data.ctrl[:N_JOINTS] = q_sol
                else:
                    # No hand detected — slowly return to home
                    q_current = self.data.qpos[:N_JOINTS].copy()
                    self.data.ctrl[:N_JOINTS] = 0.99 * q_current + 0.01 * self._home

                # Step physics
                for _ in range(8):  # 60Hz control, 0.002s timestep
                    mujoco.mj_step(self.model, self.data)

                # Render
                viewer.sync()

                step_count += 1
                if step_count % 180 == 0:  # ~ every 3 seconds
                    with self._lock:
                        t = self._target_pos
                    with self._cam_lock:
                        cm = self._cam_mode
                    if t is not None:
                        ee = self.data.site_xpos[self.site_id]
                        err = np.linalg.norm(t - ee)
                        print(f"[sim] cam={cm}  "
                              f"target={[round(x,3) for x in t]}  "
                              f"ee={[round(x,3) for x in ee]}  err={err:.3f}m")

                time.sleep(0.001)  # yield to viewer

            self._viewer = None
            print("[sim] Viewer closed")

    def stop(self):
        self._running = False
        if hasattr(self, '_sock'):
            self._sock.close()


# ── Entry point ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EGO Teleop Simulator")
    parser.add_argument("--port", type=int, default=9999, help="UDP port")
    parser.add_argument("--model", default=MODEL, help="MJCF model path")
    args = parser.parse_args()

    sim = EgoSimulator(args.model)
    sim.start_udp(args.port)

    try:
        sim.run()
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()
        print("[sim] Stopped")


if __name__ == "__main__":
    main()

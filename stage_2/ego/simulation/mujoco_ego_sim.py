#!/usr/bin/env python3
"""
EGO Teleop Simulator — Single self-contained MuJoCo simulation.

Receives hand tracking data via UDP and drives the xArm6 model with gripper.
Uses 6-DOF IK (position + orientation) via MujocoIK.
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
from mujoco_ik import MujocoIK

# ── Model ──────────────────────────────────────────────────────────────
MODEL = os.path.join(os.path.dirname(__file__), "xarm6_gripper.xml")
N_JOINTS = 6
HOME = np.array([0.0, -0.6109, -0.6981, 0.0, 1.3788, 0.0])  # rad
GRIPPER_MAX = 0.85  # drive_joint max opening (rad)


class EgoSimulator:
    """MuJoCo simulation of xArm6 + gripper driven by hand-tracking UDP data."""

    def __init__(self, model_path=MODEL):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Site and body for EE tracking
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.body_id = self.model.site_bodyid[self.site_id]

        # Gripper actuator
        self._gripper_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if self._gripper_id < 0:
            print("[sim] WARNING: 'gripper' actuator not found")

        # Init to home pose
        self.data.qpos[:N_JOINTS] = HOME
        self.data.ctrl[:N_JOINTS] = HOME
        mujoco.mj_forward(self.model, self.data)

        # IK solver (6-DOF, weighted SE(3) error)
        self._ik = MujocoIK(model_path, site_name="ee")
        self._ik.q_last = HOME.copy()

        # Print initial EE pose for reference
        home_pos, home_rot = self._ik.fk(HOME)
        print(f"[sim] Model loaded — {self._ik.n_joints} arm joints")
        print(f"[sim] HOME pose: {np.array2string(np.degrees(HOME), precision=1)} deg")
        print(f"[sim] HOME EE pos: {np.array2string(home_pos, precision=3)}")
        print(f"[sim] Gripper: {GRIPPER_MAX:.2f} rad max opening")
        print(f"[sim] IK: weighted SE(3) error (pos=45x, rot=5x), "
              f"trust={0.3}rad, smooth={0.05}, reg=[5,1,1,5,1,1]")

        # Latest hand target (thread-safe)
        self._lock = threading.Lock()
        self._target_pos = None   # [x, y, z] in robot workspace
        self._target_quat = None  # (qx, qy, qz, qw) EE orientation
        self._target_gripper = 0.0  # 0=closed, 1=open
        self._running = True

        # Camera control
        self._cam_ego_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
        self._cam_fixed_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed")
        self._cam_mode = "ego"
        self._cam_lock = threading.Lock()
        self._viewer = None

        if self._cam_ego_id < 0:
            print("[sim] WARNING: 'ego' camera not found")
        if self._cam_fixed_id < 0:
            print("[sim] WARNING: 'fixed' camera not found")

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
                gripper = msg.get("gripper", 0.0)
                with self._lock:
                    self._target_pos = np.array(wrist[:3], dtype=np.float64)
                    self._target_quat = tuple(wrist[3:7])  # qx,qy,qz,qw
                    self._target_gripper = float(gripper)
            except socket.timeout:
                pass
            except Exception:
                pass

    # ── Camera switching ───────────────────────────────────────────
    def _on_viewer_key(self, keycode: int):
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
        if self._viewer is not None:
            self._apply_camera_mode()

    def _apply_camera_mode(self):
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
        else:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # ── Main loop ──────────────────────────────────────────────────
    def run(self):
        """Open viewer and run simulation loop."""
        print(f"[sim] Opening xArm6 + gripper viewer...")
        with launch_passive(
            self.model, self.data,
            key_callback=self._on_viewer_key,
        ) as viewer:
            self._viewer = viewer
            self._apply_camera_mode()
            print(f"[sim] Camera: EGO (EE follow) — "
                  f"press 1=Ego  2=Fixed  3/Space=Free")
            print(f"[sim] Running — move your hand in front of camera")
            step_count = 0
            while self._running and viewer.is_running():
                with self._lock:
                    target_pos = self._target_pos
                    target_quat = self._target_quat
                    target_gripper = self._target_gripper

                if target_pos is not None:
                    # 6-DOF IK with orientation target
                    q_current = self.data.qpos[:N_JOINTS].copy()
                    q_sol = self._ik.solve(
                        target_pos=target_pos,
                        target_quat=target_quat,
                        q_init=q_current,
                        q_nominal=HOME,
                    )
                    if q_sol is not None:
                        self.data.ctrl[:N_JOINTS] = q_sol
                    # Gripper
                    if self._gripper_id >= 0:
                        self.data.ctrl[self._gripper_id] = (
                            target_gripper * GRIPPER_MAX)
                else:
                    # No hand / grip released — hold current position
                    self.data.ctrl[:N_JOINTS] = (
                        self.data.qpos[:N_JOINTS].copy())

                # Step physics
                for _ in range(8):
                    mujoco.mj_step(self.model, self.data)

                viewer.sync()
                step_count += 1
                if step_count % 180 == 0:
                    with self._lock:
                        tp = self._target_pos
                    with self._cam_lock:
                        cm = self._cam_mode
                    if tp is not None:
                        ee = self.data.site_xpos[self.site_id]
                        err = np.linalg.norm(tp - ee)
                        print(f"[sim] cam={cm}  "
                              f"target={[round(x,3) for x in tp]}  "
                              f"ee={[round(x,3) for x in ee]}  err={err:.3f}m")

                time.sleep(0.001)

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

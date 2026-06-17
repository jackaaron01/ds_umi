#!/usr/bin/env python3
"""
RealSense + MediaPipe Hands — EGO hand tracking with skeleton overlay.

Features:
  - Depth-based true 3D wrist position (RealSense depth map)
  - Incremental grip mode (Space to lock origin, like VR teleop)
  - Camera-to-robot extrinsic calibration
  - Multi-camera confidence fusion
  - Palm orientation + gripper control

Usage:
  python stage_2/ego/mediapipe_ego.py --udp
  python stage_2/ego/mediapipe_ego.py --udp --cam-pos 0.5 0.1 0.4 --cam-rpy 0 0 0
"""
import sys, os, time, argparse, socket, json as json_mod, copy, threading
from dataclasses import dataclass, field
import numpy as np
import cv2
import h5py

try:
    import mediapipe as mp
except ImportError:
    print("Install: pip install mediapipe")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("Install: pip install pyrealsense2")
    sys.exit(1)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── Helpers ─────────────────────────────────────────────────────────

def _extract_confidence(handedness) -> float:
    if handedness is None:
        return 0.0
    try:
        return float(handedness.classifications[0].score)
    except AttributeError:
        try:
            return float(handedness[0].classification[0].score)
        except (TypeError, IndexError, AttributeError):
            return 0.0


def _compute_palm_orientation(landmarks) -> tuple:
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
    pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    middle_mcp = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    x_axis = pinky_mcp - index_mcp
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    y_axis = middle_mcp - wrist
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    from scipy.spatial.transform import Rotation
    quat = Rotation.from_matrix(rot_matrix).as_quat()
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _compute_gripper_openness(landmarks) -> float:
    thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    index_tip = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    dist = np.linalg.norm(thumb_tip - index_tip)
    return float(np.clip((dist - 0.02) / 0.13, 0.0, 1.0))


# ── Camera-to-Robot Transform ────────────────────────────────────────

def _build_cam_to_robot(pos_m: list, rpy_deg: list) -> np.ndarray:
    """Build 4x4 transform from camera frame to robot base frame.

    Camera frame (RealSense):  +X right, +Y down, +Z forward
    Robot base frame:          +X forward, +Y left, +Z up

    Default: camera on table 0.6m in front of robot, looking at user.
    rpy rotates the camera BEFORE translation.
    """
    from scipy.spatial.transform import Rotation
    rpy_rad = np.deg2rad(rpy_deg)
    R_cam2world = Rotation.from_euler('xyz', rpy_rad).as_matrix()

    # Permute axes: camera (right,down,forward) -> world (X,Y,Z)
    # camera +Z (forward) -> world +X (forward from base)
    # camera +X (right)   -> world -Y (left from base, since robot +Y = left)
    # camera +Y (down)    -> world -Z (down)
    R_perm = np.array([
        [ 0,  0,  1],   # cam Z -> world X
        [-1,  0,  0],   # cam X -> world Y (negated: right -> left)
        [ 0, -1,  0],   # cam Y -> world Z (negated: down -> up)
    ])
    R = R_cam2world @ R_perm

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos_m
    return T


# ── Camera Capture (threaded, with depth support) ────────────────────

@dataclass
class CameraResult:
    serial: str = ""
    rgb_display: np.ndarray = field(
        default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8))
    hand_keypoints: list = field(default_factory=list)
    wrist_cam: tuple = None     # (x, y, z) in camera frame (meters)
    hand_wrist_quat: tuple = None
    hand_gripper: float = 0.0
    hand_confidence: float = 0.0
    hand_label: str = ""
    fps: float = 0.0


class CameraCapture:
    """Runs a single RealSense + MediaPipe pipeline in a background thread."""

    def __init__(self, serial: str, cam_index: int,
                 max_hands: int = 1, use_depth: bool = True):
        self.serial = serial
        self.cam_index = cam_index
        self.max_hands = max_hands
        self.use_depth = use_depth
        self._lock = threading.Lock()
        self._result = CameraResult(serial=serial)
        self._running = False
        self._thread = None

    @property
    def result(self) -> CameraResult:
        with self._lock:
            return copy.deepcopy(self._result)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if hasattr(self, '_pipeline'):
            try: self._pipeline.stop()
            except Exception: pass
        if hasattr(self, '_hands'):
            try: self._hands.close()
            except Exception: pass

    def _run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if self.use_depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipeline.start(config)
        self._pipeline = pipeline

        # Get depth intrinsics for deprojection
        depth_intrinsics = None
        if self.use_depth:
            depth_profile = profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            align = rs.align(rs.stream.color)

        hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=self.max_hands,
            min_detection_confidence=0.3, min_tracking_confidence=0.3)
        self._hands = hands

        t_last = time.time()
        while self._running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue

            # Align depth to color if available
            if self.use_depth and depth_intrinsics is not None:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            rgb_image = np.asanyarray(color_frame.get_data())
            rgb_display = rgb_image.copy()
            depth_image = None
            if self.use_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())

            rgb_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_rgb)

            hand_keypoints = []
            best_confidence = 0.0
            best_label = ""
            wrist_cam = None
            best_kp = []
            palm_quat = (0.0, 0.0, 0.0, 1.0)
            gripper_open = 0.0

            if results.multi_hand_landmarks:
                for i, hlm in enumerate(results.multi_hand_landmarks):
                    hd = results.multi_handedness[i] if results.multi_handedness else None
                    rgb_display = draw_hand_landmarks(rgb_display, hlm, hd)
                    kp_list = []
                    for lm in hlm.landmark:
                        kp_list.extend([lm.x, lm.y, lm.z])
                    hand_keypoints.append(kp_list)

                    conf = _extract_confidence(hd)
                    if conf >= best_confidence:
                        best_confidence = conf
                        best_kp = kp_list
                        try:
                            label = hd.classifications[0].label
                        except AttributeError:
                            try:
                                label = hd[0].classification[0].label
                            except (TypeError, IndexError, AttributeError):
                                label = "?"
                        best_label = label

                        # ── Depth-based 3D wrist position ──
                        w_lm = hlm.landmark[0]
                        u = int(np.clip(w_lm.x * 639, 0, 639))
                        v = int(np.clip(w_lm.y * 479, 0, 479))
                        if depth_image is not None and depth_intrinsics is not None and depth_image[v, u] > 0:
                            depth_mm = float(depth_image[v, u])
                            pt = rs.rs2_deproject_pixel_to_point(
                                depth_intrinsics, [u, v], depth_mm)
                            # pt = [x, y, z] in camera frame (meters), z=depth
                            wrist_cam = (pt[0], pt[1], pt[2])
                        else:
                            # Fallback: MediaPipe estimated z
                            wrist_cam = (
                                (w_lm.x - 0.5) * 0.6,   # ±0.3m X
                                (w_lm.y - 0.5) * 0.4,   # ±0.2m Y
                                w_lm.z * 0.8 + 0.2,     # 0.2-1.0m Z
                            )

                        palm_quat = _compute_palm_orientation(hlm.landmark)
                        gripper_open = _compute_gripper_openness(hlm.landmark)

            fps = 1.0 / max(time.time() - t_last, 0.001)
            t_last = time.time()

            if hand_keypoints:
                label_str = f"Cam{self.cam_index} {best_label}({best_confidence:.2f})" \
                            f" FPS:{fps:.0f}"
                if wrist_cam:
                    label_str += f" 3D:[{wrist_cam[0]:.2f},{wrist_cam[1]:.2f},{wrist_cam[2]:.2f}]"
                cv2.putText(rgb_display, label_str, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                hw = rgb_display.shape[1] // 2
                cv2.putText(rgb_display, f"Cam{self.cam_index} NO HAND",
                            (hw - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            with self._lock:
                self._result = CameraResult(
                    serial=self.serial, rgb_display=rgb_display,
                    hand_keypoints=hand_keypoints, wrist_cam=wrist_cam,
                    hand_wrist_quat=palm_quat, hand_gripper=gripper_open,
                    hand_confidence=best_confidence, hand_label=best_label,
                    fps=fps)


# ── Globals for key handling ─────────────────────────────────────────

mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
PROJ = os.path.dirname(os.path.abspath(__file__))
_quit = False
_grip_locked = False
_grip_origin = None  # [x, y, z] in robot frame, recorded at grip press


def _on_key(event):
    global _quit, _grip_locked, _grip_origin
    if event.key == 'q':
        _quit = True
    elif event.key == ' ':
        if not _grip_locked:
            _grip_locked = True
            print("  [GRIP] LOCK — press Space again to release")
        else:
            _grip_locked = False
            _grip_origin = None
            print("  [GRIP] RELEASED")


def draw_hand_landmarks(image, hand_landmarks, handedness=None):
    h, w, _ = image.shape
    for conn in HAND_CONNECTIONS:
        s_idx, e_idx = conn
        s_lm = hand_landmarks.landmark[s_idx]
        e_lm = hand_landmarks.landmark[e_idx]
        x1, y1 = int(s_lm.x * w), int(s_lm.y * h)
        x2, y2 = int(e_lm.x * w), int(e_lm.y * h)
        if s_idx <= 4 and e_idx <= 4:
            color = (0, 255, 0)
        elif 5 <= s_idx <= 8 and 5 <= e_idx <= 8:
            color = (255, 0, 0)
        else:
            color = (255, 255, 255)
        cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)
        if idx == 0:
            color, radius = (0, 255, 255), 6
        elif idx in [4, 8, 12, 16, 20]:
            color, radius = (0, 0, 255), 5
        else:
            color, radius = (200, 200, 200), 3
        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)
    if handedness is not None:
        try:
            label = handedness.classifications[0].label
            score = handedness.classifications[0].score
        except AttributeError:
            try:
                label = handedness[0].classification[0].label
                score = handedness[0].classification[0].score
            except (TypeError, IndexError, AttributeError):
                label, score = "?", 0.0
        cv2.putText(image, f"{label} ({score:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global _quit, _grip_locked, _grip_origin
    parser = argparse.ArgumentParser(description="EGO Hand Tracking")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--udp", action="store_true")
    parser.add_argument("--udp-port", type=int, default=9999)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--camera-serials", type=str, nargs="*", default=None)
    parser.add_argument("--camera-ids", type=int, nargs="*", default=None)
    # Calibration: camera position in robot base frame (meters)
    parser.add_argument("--cam-pos", type=float, nargs=3,
                        default=[0.6, 0.0, 0.3],
                        help="Camera position [X,Y,Z] in robot base frame (m)")
    parser.add_argument("--cam-rpy", type=float, nargs=3,
                        default=[0, 0, 0],
                        help="Camera RPY rotation [roll,pitch,yaw] (degrees)")
    # Incremental mode
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Position scaling factor")
    parser.add_argument("--no-incremental", action="store_true",
                        help="Disable incremental mode (use absolute mapping)")
    args = parser.parse_args()

    OUT = args.output or os.path.join(PROJ, "..", "data", "mediapipe_ego.h5")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    # ── Camera enumeration ──
    camera_serials = []
    if args.camera_serials:
        camera_serials = list(args.camera_serials)
    elif args.camera_ids:
        ctx = rs.context()
        all_devices = ctx.query_devices()
        for idx in args.camera_ids:
            if idx < len(all_devices):
                sn = all_devices[idx].get_info(rs.camera_info.serial_number)
                camera_serials.append(sn)
        if not camera_serials:
            print("ERROR: No valid cameras for --camera-ids"); sys.exit(1)
    else:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("ERROR: No RealSense cameras detected"); sys.exit(1)
        camera_serials = [devices[0].get_info(rs.camera_info.serial_number)]
    n_cams = len(camera_serials)
    print(f"Cameras ({n_cams}): {camera_serials}")

    # ── Calibration ──
    T_cam2robot = _build_cam_to_robot(args.cam_pos, args.cam_rpy)
    print(f"Calibration: pos={args.cam_pos} rpy={args.cam_rpy}°")
    print(f"  T_cam2robot:\n{T_cam2robot}")
    print(f"  Incremental mode: {'OFF' if args.no_incremental else 'ON (Space to grip)'}")
    print(f"  Scale: {args.scale}")

    # ── Launch capture threads ──
    captures: list = []
    for i, sn in enumerate(camera_serials):
        cap = CameraCapture(serial=sn, cam_index=i,
                            max_hands=args.max_hands,
                            use_depth=not args.no_depth)
        cap.start()
        captures.append(cap)
    time.sleep(1.0)

    # ── UDP ──
    udp_sock = None
    if args.udp:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"  UDP: → Docker:{args.udp_port}")

    # ── Recording ──
    recording = args.record
    frames_rgb, keypoints_data, timestamps = [], [], []
    frame_count, total_frames = 0, 0

    print(f"\n{'='*55}")
    print(f"  MediaPipe EGO | Cams:{n_cams} | "
          f"UDP:{'ON' if args.udp else 'OFF'} | "
          f"Rec:{'ON' if recording else 'OFF'}")
    print(f"  Space = Grip lock | q = Quit")
    if args.no_incremental:
        print(f"  *** Absolute mapping mode (no grip) ***")
    print(f"{'='*55}")

    # ── Matplotlib ──
    plt.ion()
    figsize = {1: (9, 5), 2: (16, 5)}.get(n_cams, (14, 10))
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    img_handle = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    ax.set_title(f"MediaPipe EGO — {n_cams} Camera(s)")
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)

    try:
        while not _quit:
            all_results = [cap.result for cap in captures]
            best = max(all_results, key=lambda r: r.hand_confidence)

            # ── Tiled display ──
            if n_cams == 1:
                rgb_display = all_results[0].rgb_display
            elif n_cams == 2:
                rgb_display = np.hstack([r.rgb_display for r in all_results])
            else:
                cols = min(n_cams, 3)
                rows = (n_cams + cols - 1) // cols
                blank = np.zeros_like(all_results[0].rgb_display)
                rgb_display = np.vstack([
                    np.hstack([
                        all_results[r * cols + c].rgb_display
                        if r * cols + c < n_cams else blank
                        for c in range(cols)
                    ]) for r in range(rows)
                ])

            # ── Transform + UDP ──
            if udp_sock and best.wrist_cam is not None:
                # Camera → Robot transform
                cx, cy, cz = best.wrist_cam
                cam_pos = np.array([cx, cy, cz, 1.0])
                robot_pos = (T_cam2robot @ cam_pos)[:3]

                # Incremental mode
                if not args.no_incremental:
                    if _grip_locked:
                        if _grip_origin is None:
                            _grip_origin = robot_pos.copy()
                            print(f"  [GRIP] Origin locked: "
                                  f"[{robot_pos[0]:.3f},{robot_pos[1]:.3f},{robot_pos[2]:.3f}]")
                        delta = robot_pos - _grip_origin
                        rx = delta[0] * args.scale
                        ry = delta[1] * args.scale
                        rz = delta[2] * args.scale
                    else:
                        # No grip — no UDP (robot stays)
                        rx = ry = rz = None
                else:
                    # Absolute mode
                    rx, ry, rz = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])

                if rx is not None:
                    qx, qy, qz, qw = best.hand_wrist_quat or (0, 0, 0, 1)
                    kp = best.hand_keypoints[0] if best.hand_keypoints else []
                    udp_data = json_mod.dumps({
                        "wrist": [rx, ry, rz, qx, qy, qz, qw],
                        "keypoints": kp,
                        "gripper": best.hand_gripper,
                        "confidence": best.hand_confidence,
                        "camera": best.serial,
                    })
                    udp_sock.sendto(udp_data.encode(), ("127.0.0.1", args.udp_port))

            # ── Overlay ──
            fps_avg = np.mean([r.fps for r in all_results])
            status = "[REC]" if recording else "[LIVE]"
            hands_total = sum(len(r.hand_keypoints) for r in all_results)
            grip_str = "[GRIP]" if _grip_locked else "[FREE]"
            cv2.putText(rgb_display,
                        f"{status} {grip_str} Cams:{n_cams} FPS:{fps_avg:.0f} "
                        f"Hands:{hands_total} Best:{best.hand_label}"
                        f"({best.hand_confidence:.2f})",
                        (10, rgb_display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_handle.set_data(cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB))
            ax.set_title(f"MediaPipe EGO | {n_cams} cam | {grip_str} | "
                         f"FPS:{fps_avg:.0f} | Hands:{hands_total}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            total_frames += 1
            if total_frames % 100 == 0:
                per_cam = " | ".join(
                    f"cam{i}:{r.hand_label}({r.hand_confidence:.2f})"
                    for i, r in enumerate(all_results))
                print(f"  [frame {total_frames}] FPS:{fps_avg:.0f} | "
                      f"Hands:{hands_total} | {grip_str} | {per_cam}")

            if recording:
                frames_rgb.append(best.rgb_display.copy())
                keypoints_data.append(
                    best.hand_keypoints[0] if best.hand_keypoints else [])
                timestamps.append(time.time())
                frame_count += 1

    finally:
        for cap in captures:
            cap.stop()
        if udp_sock:
            udp_sock.close()
        plt.close("all")
        if recording and frames_rgb:
            rgb = np.stack(frames_rgb, 0)
            ts = np.array(timestamps, dtype=np.float64)
            with h5py.File(OUT, "w") as f:
                ep = f.create_group("episode_000000")
                ep.create_dataset("sensors/camera/ego_rgb", data=rgb,
                                  compression="gzip", chunks=(1, *rgb.shape[1:]))
                ep.create_dataset("timestamp", data=ts)
                ep.attrs["num_frames"] = len(rgb)
            print(f"  Saved: {OUT} ({len(rgb)} frames)")


if __name__ == "__main__":
    main()

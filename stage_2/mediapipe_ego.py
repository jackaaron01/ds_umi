#!/usr/bin/env python3
"""
RealSense + MediaPipe Hands — EGO hand tracking with skeleton overlay.

Captures from RealSense D435i, runs MediaPipe Hands, displays hand skeleton
overlay in real-time, and records synchronized camera + hand keypoints.

Usage (on HOST):
    pip install mediapipe pyrealsense2 opencv-python h5py
    python3 stage_2/mediapipe_ego.py                    # debug view only
    python3 stage_2/mediapipe_ego.py --record           # record to HDF5
    python3 stage_2/mediapipe_ego.py --record --output data/my_ego.h5
"""
import sys, os, time, argparse, socket, json as json_mod
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

# ═══════════════════════════════════════════════════════════════
# MediaPipe Hands setup
# ═══════════════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand landmark connections for skeleton drawing
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Landmark names for reference
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

PROJ = os.path.dirname(os.path.abspath(__file__))


def draw_hand_landmarks(image, hand_landmarks, handedness=None):
    """Draw hand skeleton with styled connections and landmark dots."""
    h, w, _ = image.shape

    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start = hand_landmarks.landmark[start_idx]
        end = hand_landmarks.landmark[end_idx]

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)

        # Thumb = green, index = blue, others = white
        if start_idx <= 4 and end_idx <= 4:
            color = (0, 255, 0)  # green for thumb
        elif 5 <= start_idx <= 8 and 5 <= end_idx <= 8:
            color = (255, 0, 0)  # blue for index
        else:
            color = (255, 255, 255)  # white for others

        cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Draw landmark points
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)

        # Color by landmark type
        if idx == 0:  # wrist
            color = (0, 255, 255)  # yellow
            radius = 6
        elif idx in [4, 8, 12, 16, 20]:  # fingertips
            color = (0, 0, 255)  # red
            radius = 5
        else:
            color = (200, 200, 200)
            radius = 3

        cv2.circle(image, (x, y), radius, color, -1, cv2.LINE_AA)

    # Handedness label
    if handedness:
        try:
            label = handedness[0].classification[0].label
            score = handedness[0].classification[0].score
        except (TypeError, IndexError):
            # ClassificationList directly (not in list)
            try:
                label = handedness.classification[0].label
                score = handedness.classification[0].score
            except AttributeError:
                label, score = "?", 0.0
        cv2.putText(image, f"{label} ({score:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image


def get_fingertip_positions(hand_landmarks, image_shape):
    """Extract fingertip 3D positions (normalized + depth if available)."""
    h, w = image_shape[:2]
    tips = {}
    for name, idx in [("THUMB", 4), ("INDEX", 8), ("MIDDLE", 12),
                       ("RING", 16), ("PINKY", 20)]:
        lm = hand_landmarks.landmark[idx]
        tips[name] = np.array([lm.x * w, lm.y * h, lm.z * w])  # pixel coords
    # Wrist
    wrist = hand_landmarks.landmark[0]
    tips["WRIST"] = np.array([wrist.x * w, wrist.y * h, wrist.z * w])
    return tips


def compute_gripper_from_fingers(thumb_tip, index_tip):
    """Compute gripper command (0-1) from thumb-index distance."""
    dist = np.linalg.norm(thumb_tip - index_tip)
    close_thresh, open_thresh = 20, 80
    gripper = 1.0 - np.clip((dist - close_thresh) / (open_thresh - close_thresh), 0, 1)
    return float(gripper)


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hand EGO Tracker")
    parser.add_argument("--record", action="store_true", help="Record to HDF5")
    parser.add_argument("--output", default=None,
                        help="Output HDF5 path (default: data/mediapipe_ego.h5)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Skip depth stream (faster)")
    parser.add_argument("--max-hands", type=int, default=1,
                        help="Max hands to detect")
    parser.add_argument("--udp", action="store_true",
                        help="Send hand keypoints via UDP to Docker ROS2 bridge")
    parser.add_argument("--udp-port", type=int, default=9999,
                        help="UDP port for ROS2 bridge")
    args = parser.parse_args()

    OUT = args.output or os.path.join(PROJ, "..", "data", "mediapipe_ego.h5")

    # ── RealSense pipeline ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Starting RealSense...")
    profile = pipeline.start(config)

    # Get depth scale
    depth_scale = 0.001
    if not args.no_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"  Depth scale: {depth_scale:.4f}m")

    # Align depth to color
    align = rs.align(rs.stream.color) if not args.no_depth else None

    # ── MediaPipe Hands ──
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_hands,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # ── Recording state ──
    recording = args.record
    frames_rgb, frames_depth = [], []
    keypoints_data = []
    timestamps = []
    frame_count = 0
    t0 = time.time()

    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    # ── UDP sender (to Docker ROS2 bridge) ──
    udp_sock = None
    if args.udp:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"  UDP: sending to localhost:{args.udp_port} → Docker ROS2 bridge")

    print(f"\n{'='*55}")
    print(f"  MediaPipe EGO Tracker")
    print(f"  Max hands: {args.max_hands}  |  Depth: {not args.no_depth}")
    print(f"  UDP to Docker: {'ON (port '+str(args.udp_port)+')' if args.udp else 'OFF'}")
    print(f"  Recording: {'ON → ' + OUT if recording else 'OFF'}")
    print(f"  'r'=record  's'=snapshot  'q'=quit")
    print(f"{'='*55}")

    try:
        while True:
            # ── Capture ──
            frames = pipeline.wait_for_frames()
            if align:
                frames = align.process(frames)

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            rgb_image = np.asanyarray(color_frame.get_data())
            rgb_display = rgb_image.copy()

            depth_image = None
            if not args.no_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())

            # ── MediaPipe processing ──
            rgb_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_rgb)

            # ── Visualize hand skeleton ──
            hand_keypoints = []
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[i] if results.multi_handedness else None
                    rgb_display = draw_hand_landmarks(rgb_display, hand_landmarks, handedness)

                    # Extract keypoints
                    kp_list = []
                    for lm in hand_landmarks.landmark:
                        kp_list.extend([lm.x, lm.y, lm.z])
                    hand_keypoints.append(kp_list)

                    # ── Send via UDP to Docker ROS2 bridge ──
                    if udp_sock and hand_landmarks:
                        wrist_lm = hand_landmarks.landmark[0]
                        # Convert normalized coords to approximate meters
                        # Use depth at wrist pixel for Z, or default scale
                        wrist_x = (wrist_lm.x - 0.5) * 0.5  # rough meter conversion
                        wrist_y = (wrist_lm.y - 0.5) * 0.5
                        wrist_z = wrist_lm.z * 0.5 + 0.3

                        udp_data = json_mod.dumps({
                            "wrist": [wrist_x, wrist_y, wrist_z, 0, 0, 0, 1],
                            "keypoints": kp_list,
                        })
                        udp_sock.sendto(udp_data.encode("utf-8"),
                                        ("127.0.0.1", args.udp_port))

                    # Show fingertip info
                    tips = get_fingertip_positions(hand_landmarks, rgb_display.shape)
                    if "THUMB" in tips and "INDEX" in tips:
                        gripper = compute_gripper_from_fingers(
                            tips["THUMB"], tips["INDEX"])
                        cv2.putText(rgb_display, f"Gripper: {gripper:.2f}",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)

            # ── FPS display ──
            fps = 1.0 / max(time.time() - t0, 0.001)
            t0 = time.time()
            status = "[REC]" if recording else "[LIVE]"
            cv2.putText(rgb_display, f"{status} FPS:{fps:.0f} Frames:{frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if recording:
                cv2.circle(rgb_display, (rgb_display.shape[1] - 30, 30),
                           10, (0, 0, 255), -1)

            # ── Depth visualization ──
            if depth_image is not None:
                depth_cmap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03),
                    cv2.COLORMAP_JET)
                display = np.hstack((rgb_display, depth_cmap))
            else:
                display = rgb_display

            cv2.imshow("MediaPipe EGO — Hand Tracking", display)
            key = cv2.waitKey(1) & 0xFF

            # ── Keyboard controls ──
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print(f"  RECORDING STARTED → {OUT}")
                    frames_rgb, frames_depth = [], []
                    keypoints_data, timestamps = [], []
                    frame_count = 0
                else:
                    _save_recording(OUT, frames_rgb, frames_depth,
                                    keypoints_data, timestamps, depth_scale)
                    frames_rgb, frames_depth = [], []
                    keypoints_data, timestamps = [], []
                    frame_count = 0
            elif key == ord('s'):
                out_dir = os.path.join(PROJ, "..", "outputs", "figures")
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, "mediapipe_snapshot.png"), rgb_display)
                print(f"  Snapshot saved to outputs/figures/")

            # ── Record ──
            if recording:
                frames_rgb.append(rgb_image.copy())
                if depth_image is not None:
                    frames_depth.append(depth_image.copy())
                keypoints_data.append(hand_keypoints)
                timestamps.append(time.time())
                frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        hands.close()
        if udp_sock:
            udp_sock.close()

        if recording and frames_rgb:
            _save_recording(OUT, frames_rgb, frames_depth,
                            keypoints_data, timestamps, depth_scale)


def _save_recording(path, rgb_list, depth_list, kp_list, ts_list, depth_scale):
    """Save recorded data to HDF5."""
    if not rgb_list:
        return

    rgb = np.stack(rgb_list, 0)
    ts = np.array(ts_list, dtype=np.float64)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with h5py.File(path, "w") as f:
        ep = f.create_group("episode_000000")
        ep.create_dataset("sensors/camera/ego_rgb", data=rgb,
                          compression="gzip", chunks=(1, *rgb.shape[1:]))
        if depth_list:
            depth = np.stack(depth_list, 0)
            ep.create_dataset("sensors/camera/ego_depth", data=depth,
                              compression="gzip", chunks=(1, *depth.shape[1:]))

        # Store hand keypoints as variable-length (max 2 hands, 21 landmarks × 3)
        max_kp = max(len(k) for k in kp_list) if any(kp_list) else 0
        if max_kp > 0:
            # Pad all keypoint lists to same length
            kp_padded = np.zeros((len(kp_list), max(max_kp, 1), 63), dtype=np.float32)
            for i, hands_kp in enumerate(kp_list):
                for j, hand_kp in enumerate(hands_kp[:max_kp]):
                    if len(hand_kp) <= 63:
                        kp_padded[i, j, :len(hand_kp)] = hand_kp
            ep.create_dataset("hand_keypoints", data=kp_padded, compression="gzip")

        ep.create_dataset("timestamp", data=ts)
        ep.attrs["num_frames"] = len(rgb)
        ep.attrs["depth_scale"] = depth_scale

    size_mb = os.path.getsize(path) / 1024**2
    print(f"  ✓ Saved: {path} ({len(rgb)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

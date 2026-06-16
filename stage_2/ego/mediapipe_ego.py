#!/usr/bin/env python3
"""
RealSense + MediaPipe Hands — EGO hand tracking with skeleton overlay.

Displays via matplotlib (works in conda without Qt/GTK issues).
Usage: python stage_2/ego/mediapipe_ego.py --udp
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

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ═══════════════════════════════════════════════════════════════
mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
PROJ = os.path.dirname(os.path.abspath(__file__))

# Track 'q' key to quit
_quit = False


def _on_key(event):
    global _quit
    if event.key == 'q':
        _quit = True
    elif event.key == 'r':
        _quit = False  # will be handled in main loop
    elif event.key == 's':
        pass  # handled in main loop


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
            # MediaPipe 0.10+: ClassificationList.classifications (plural)
            label = handedness.classifications[0].label
            score = handedness.classifications[0].score
        except AttributeError:
            try:
                # Older MediaPipe API: list of Classification objects
                label = handedness[0].classification[0].label
                score = handedness[0].classification[0].score
            except (TypeError, IndexError, AttributeError):
                label, score = "?", 0.0
        cv2.putText(image, f"{label} ({score:.2f})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return image


def main():
    global _quit
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--udp", action="store_true")
    parser.add_argument("--udp-port", type=int, default=9999)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument("--no-depth", action="store_true")
    args = parser.parse_args()

    OUT = args.output or os.path.join(PROJ, "..", "data", "mediapipe_ego.h5")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    # ── RealSense ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    print("Starting RealSense...")
    profile = pipeline.start(config)
    depth_scale = 0.001
    if not args.no_depth:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    print(f"  Depth scale: {depth_scale:.4f}m")

    # ── MediaPipe ──
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=args.max_hands,
                           min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # ── UDP ──
    udp_sock = None
    if args.udp:
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"  UDP: → Docker:{args.udp_port}")

    # ── Recording state ──
    recording = args.record
    frames_rgb, frames_depth, keypoints_data, timestamps = [], [], [], []
    frame_count, total_frames = 0, 0
    t0_rec = time.time()
    last_key = None

    print(f"\n{'='*50}")
    print(f"  MediaPipe EGO | UDP:{'ON' if args.udp else 'OFF'} | "
          f"Record:{'ON' if recording else 'OFF'}")
    print(f"  Close window or press 'q' to quit")
    print(f"{'='*50}")

    # ── Matplotlib figure ──
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.canvas.mpl_connect('key_press_event', _on_key)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_handle = ax.imshow(dummy_img)
    ax.set_title("MediaPipe EGO — Hand Tracking")
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)

    try:
        while not _quit:
            # ── Capture ──
            frames = pipeline.wait_for_frames()
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

            # ── MediaPipe ──
            rgb_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_rgb)

            hand_keypoints = []
            if results.multi_hand_landmarks:
                for i, hlm in enumerate(results.multi_hand_landmarks):
                    hd = results.multi_handedness[i] if results.multi_handedness else None
                    rgb_display = draw_hand_landmarks(rgb_display, hlm, hd)
                    kp_list = []
                    for lm in hlm.landmark:
                        kp_list.extend([lm.x, lm.y, lm.z])
                    hand_keypoints.append(kp_list)

                    # UDP send
                    if udp_sock:
                        w_lm = hlm.landmark[0]
                        # Map normalized hand coords → robot workspace (meters)
                        # Center the mapping around the EE home position
                        # Hand X (left-right, 0-1)  → robot Y (-0.2 to 0.2)
                        # Hand Y (up-down, 0-1)     → robot Z (0.15 to 0.45)
                        # Hand Z (depth, near-far)  → robot X (0.05 to 0.35)
                        ry = (0.5 - w_lm.x) * 0.4           # ±0.2m
                        rz = (1.0 - w_lm.y) * 0.3 + 0.15    # 0.15-0.45m
                        rx = (1.0 - w_lm.z) * 0.3 + 0.05    # 0.05-0.35m
                        udp_data = json_mod.dumps({
                            "wrist": [rx, ry, rz, 0, 0, 0, 1],
                            "keypoints": kp_list,
                        })
                        udp_sock.sendto(udp_data.encode(), ("127.0.0.1", args.udp_port))

            # ── Overlays ──
            fps = 1.0 / max(time.time() - t0_rec, 0.001)
            t0_rec = time.time()
            status = "[REC]" if recording else "[LIVE]"
            cv2.putText(rgb_display, f"{status} FPS:{fps:.0f} Hands:{len(hand_keypoints)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if not hand_keypoints:
                hw = rgb_display.shape[1] // 2
                cv2.putText(rgb_display, "NO HAND - Show hand to camera",
                            (hw - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # ── Update matplotlib display ──
            img_handle.set_data(cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB))
            ax.set_title(f"MediaPipe EGO | FPS:{fps:.0f} | Hands:{len(hand_keypoints)} | "
                         f"{status} {'UDP:ON' if udp_sock else ''}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            # ── Periodic console status ──
            total_frames += 1
            if total_frames % 100 == 0:
                print(f"  [frame {total_frames}] FPS:{fps:.0f} | Hands:{len(hand_keypoints)}")

            # ── Recording ──
            if recording:
                frames_rgb.append(rgb_image.copy())
                if depth_image is not None:
                    frames_depth.append(depth_image.copy())
                keypoints_data.append(hand_keypoints)
                timestamps.append(time.time())
                frame_count += 1

            # ── Keyboard shortcuts (read from matplotlib events) ──
            # 'r' toggle recording, 's' snapshot
            # These come through _on_key as key_press_event
            if last_key != getattr(fig, '_last_key', None):
                pass  # key handling via _on_key

    finally:
        pipeline.stop()
        hands.close()
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
                if frames_depth:
                    d = np.stack(frames_depth, 0)
                    ep.create_dataset("sensors/camera/ego_depth", data=d,
                                      compression="gzip", chunks=(1, *d.shape[1:]))
                ep.create_dataset("timestamp", data=ts)
                ep.attrs["num_frames"] = len(rgb)
            print(f"  Saved: {OUT} ({len(rgb)} frames)")


if __name__ == "__main__":
    main()

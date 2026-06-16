#!/usr/bin/env python3
"""
RealSense + MediaPipe Hands — EGO hand tracking with skeleton overlay.

Displays via matplotlib (works in conda without Qt/GTK issues).
Usage: python stage_2/ego/mediapipe_ego.py --udp
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
from matplotlib.animation import FuncAnimation

def _extract_confidence(handedness) -> float:
    """Extract confidence score from MediaPipe handedness result."""
    if handedness is None:
        return 0.0
    try:
        return float(handedness.classifications[0].score)
    except AttributeError:
        try:
            return float(handedness[0].classification[0].score)
        except (TypeError, IndexError, AttributeError):
            return 0.0


# ═══════════════════════════════════════════════════════════════
# Multi-camera support
# ═══════════════════════════════════════════════════════════════

@dataclass
class CameraResult:
    """Thread-safe result from one camera thread."""
    serial: str = ""
    rgb_display: np.ndarray = field(
        default_factory=lambda: np.zeros((480, 640, 3), dtype=np.uint8))
    hand_keypoints: list = field(default_factory=list)
    hand_wrist_xyz: tuple = None  # (rx, ry, rz) in robot workspace
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
            try:
                self._pipeline.stop()
            except Exception:
                pass
        if hasattr(self, '_hands'):
            try:
                self._hands.close()
            except Exception:
                pass

    def _run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        if self.use_depth:
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pipeline.start(config)
        self._pipeline = pipeline

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self._hands = hands

        t_last = time.time()
        while self._running:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except RuntimeError:
                continue
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            rgb_image = np.asanyarray(color_frame.get_data())
            rgb_display = rgb_image.copy()

            # MediaPipe
            rgb_rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_rgb)

            hand_keypoints = []
            best_confidence = 0.0
            best_label = ""
            wrist_xyz = None
            best_kp = []

            if results.multi_hand_landmarks:
                for i, hlm in enumerate(results.multi_hand_landmarks):
                    hd = results.multi_handedness[i] if results.multi_handedness else None
                    rgb_display = draw_hand_landmarks(rgb_display, hlm, hd)
                    kp_list = []
                    for lm in hlm.landmark:
                        kp_list.extend([lm.x, lm.y, lm.z])
                    hand_keypoints.append(kp_list)

                    conf = _extract_confidence(hd)
                    if conf > best_confidence:
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
                        w_lm = hlm.landmark[0]
                        ry = (0.5 - w_lm.x) * 0.4
                        rz = (1.0 - w_lm.y) * 0.3 + 0.15
                        rx = (1.0 - w_lm.z) * 0.3 + 0.05
                        wrist_xyz = (rx, ry, rz)

            fps = 1.0 / max(time.time() - t_last, 0.001)
            t_last = time.time()

            # Overlay status
            if hand_keypoints:
                cv2.putText(rgb_display,
                            f"Cam{self.cam_index} {best_label}({best_confidence:.2f}) FPS:{fps:.0f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
            else:
                hw = rgb_display.shape[1] // 2
                cv2.putText(rgb_display, f"Cam{self.cam_index} NO HAND",
                            (hw - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            with self._lock:
                self._result = CameraResult(
                    serial=self.serial,
                    rgb_display=rgb_display,
                    hand_keypoints=hand_keypoints,
                    hand_wrist_xyz=wrist_xyz,
                    hand_confidence=best_confidence,
                    hand_label=best_label,
                    fps=fps,
                )


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
    parser.add_argument("--camera-serials", type=str, nargs="*", default=None,
                        help="RealSense serial numbers for multi-camera")
    parser.add_argument("--camera-ids", type=int, nargs="*", default=None,
                        help="V4L2 device indices (backup, serial numbers preferred)")
    args = parser.parse_args()

    OUT = args.output or os.path.join(PROJ, "..", "data", "mediapipe_ego.h5")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

    # ── Camera enumeration ──
    camera_serials = []
    if args.camera_serials:
        camera_serials = list(args.camera_serials)
        print(f"Using {len(camera_serials)} camera(s) by serial: {camera_serials}")
    elif args.camera_ids:
        ctx = rs.context()
        all_devices = ctx.query_devices()
        for idx in args.camera_ids:
            if idx < len(all_devices):
                sn = all_devices[idx].get_info(rs.camera_info.serial_number)
                camera_serials.append(sn)
                print(f"  V4L2 index {idx} → serial {sn}")
            else:
                print(f"WARNING: V4L2 index {idx} out of range "
                      f"(found {len(all_devices)} devices)")
        if not camera_serials:
            print("ERROR: No valid cameras found for given --camera-ids")
            sys.exit(1)
    else:
        # Default: auto-detect first RealSense
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("ERROR: No RealSense cameras detected")
            sys.exit(1)
        camera_serials = [devices[0].get_info(rs.camera_info.serial_number)]
        print(f"Single camera (auto-detected): serial={camera_serials[0]}")
    n_cams = len(camera_serials)

    # ── Launch camera capture threads ──
    captures: list = []
    for i, sn in enumerate(camera_serials):
        cap = CameraCapture(
            serial=sn, cam_index=i,
            max_hands=args.max_hands,
            use_depth=not args.no_depth,
        )
        cap.start()
        captures.append(cap)
        print(f"  Camera {i}: serial={sn} started")
    time.sleep(1.0)  # let first frames arrive

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

    print(f"\n{'='*50}")
    print(f"  MediaPipe EGO | Cameras:{n_cams} | "
          f"UDP:{'ON' if args.udp else 'OFF'} | "
          f"Record:{'ON' if recording else 'OFF'}")
    print(f"  Close window or press 'q' to quit")
    print(f"{'='*50}")

    # ── Matplotlib figure ──
    plt.ion()
    if n_cams == 2:
        figsize = (16, 5)
    elif n_cams >= 3:
        figsize = (14, 10)
    else:
        figsize = (9, 5)
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.mpl_connect('key_press_event', _on_key)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_handle = ax.imshow(dummy_img)
    ax.set_title(f"MediaPipe EGO — {n_cams} Camera(s)")
    ax.axis("off")
    plt.tight_layout()
    plt.show(block=False)

    try:
        while not _quit:
            # ── Collect results from all cameras ──
            all_results = [cap.result for cap in captures]

            # Fuse: pick best by confidence
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
                row_imgs = []
                for row_idx in range(rows):
                    row_cams = []
                    for col_idx in range(cols):
                        idx = row_idx * cols + col_idx
                        if idx < n_cams:
                            row_cams.append(all_results[idx].rgb_display)
                        else:
                            row_cams.append(blank)
                    row_imgs.append(np.hstack(row_cams))
                rgb_display = np.vstack(row_imgs)

            # ── UDP send: best-confidence result ──
            if udp_sock and best.hand_wrist_xyz is not None:
                rx, ry, rz = best.hand_wrist_xyz
                kp = best.hand_keypoints[0] if best.hand_keypoints else []
                udp_data = json_mod.dumps({
                    "wrist": [rx, ry, rz, 0, 0, 0, 1],
                    "keypoints": kp,
                    "confidence": best.hand_confidence,
                    "camera": best.serial,
                })
                udp_sock.sendto(udp_data.encode(), ("127.0.0.1", args.udp_port))

            # ── Status overlay on composite ──
            fps_avg = np.mean([r.fps for r in all_results])
            status = "[REC]" if recording else "[LIVE]"
            hands_total = sum(len(r.hand_keypoints) for r in all_results)
            cv2.putText(rgb_display,
                        f"{status} Cameras:{n_cams} FPS:{fps_avg:.0f} "
                        f"Hands:{hands_total} Best:{best.hand_label}"
                        f"({best.hand_confidence:.2f})",
                        (10, rgb_display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ── Update display ──
            img_handle.set_data(cv2.cvtColor(rgb_display, cv2.COLOR_BGR2RGB))
            ax.set_title(f"MediaPipe EGO | {n_cams} cam(s) | "
                         f"FPS:{fps_avg:.0f} | Hands:{hands_total} | "
                         f"{status} {'UDP:ON' if udp_sock else ''}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            # ── Periodic console status ──
            total_frames += 1
            if total_frames % 100 == 0:
                per_cam = " | ".join(
                    f"cam{i}:{r.hand_label}({r.hand_confidence:.2f})"
                    for i, r in enumerate(all_results))
                print(f"  [frame {total_frames}] FPS:{fps_avg:.0f} | "
                      f"Hands:{hands_total} | {per_cam}")

            # ── Recording (stores best-camera frame) ──
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

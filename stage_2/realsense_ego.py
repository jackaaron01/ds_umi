#!/usr/bin/env python3
"""
RealSense EGO camera debug viewer & recorder.

Shows RGB and depth streams from the RealSense D435i.
Press 's' to save a frame, 'r' to start/stop recording, 'q' to quit.

Recordings are saved as HDF5 files (UMI format) with:
  - sensors/camera/ego_rgb
  - sensors/camera/ego_depth
  - timestamps

Usage:
    python3 realsense_ego.py                 # debug viewer
    python3 realsense_ego.py --record         # record to HDF5
    python3 realsense_ego.py --output data/ego_real.h5  # custom output
"""
import sys, os, time, argparse
import numpy as np
import cv2
import h5py

try:
    import pyrealsense2 as rs
except ImportError:
    print("pyrealsense2 not installed. Run: pip install pyrealsense2")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="RealSense EGO debug & record")
    parser.add_argument("--record", action="store_true", help="Record to HDF5")
    parser.add_argument("--output", default="/workspace/umi/data/ego_real.h5")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # ── Initialize RealSense pipeline ──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print(f"Starting RealSense pipeline ({args.width}×{args.height} @ {args.fps}fps)...")
    profile = pipeline.start(config)

    # Get depth scale for converting to meters
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"  Depth scale: {depth_scale:.4f}m")

    # Align depth to color
    align = rs.align(rs.stream.color)

    # ── Recording state ──
    recording = args.record
    rgb_frames = []
    depth_frames = []
    timestamps = []
    frame_count = 0
    t0 = time.time()

    print(f"\nControls:")
    print(f"  'r' - start/stop recording")
    print(f"  's' - save current frame as PNG")
    print(f"  'q' - quit")
    print(f"  Recording: {'ON' if recording else 'OFF'}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Normalize depth for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET)

            # Stack RGB + depth side by side
            display = np.hstack((color_image, depth_colormap))

            # Overlay info
            fps_display = 1.0 / max(time.time() - t0, 0.001)
            t0 = time.time()
            rec_status = "[REC]" if recording else "[LIVE]"
            cv2.putText(display, f"{rec_status} FPS: {fps_display:.0f} Frames: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if recording:
                cv2.circle(display, (display.shape[1] - 30, 30), 10, (0, 0, 255), -1)

            cv2.imshow("RealSense EGO (RGB + Depth)", display)
            key = cv2.waitKey(1) & 0xFF

            # ── Handle keys ──
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print(f"  Recording STARTED (frame {frame_count})")
                    rgb_frames = []
                    depth_frames = []
                    timestamps = []
                else:
                    # Save recording
                    _save_recording(args.output, rgb_frames, depth_frames,
                                    timestamps, depth_scale)
                    rgb_frames = []
                    depth_frames = []
                    timestamps = []
            elif key == ord('s'):
                cv2.imwrite("/workspace/umi/outputs/ego_snapshot_rgb.png", color_image)
                cv2.imwrite("/workspace/umi/outputs/ego_snapshot_depth.png", depth_colormap)
                print(f"  Snapshot saved to outputs/ego_snapshot_*.png")

            # ── Record if active ──
            if recording:
                rgb_frames.append(color_image.copy())
                depth_frames.append(depth_image.copy())
                timestamps.append(time.time())
                frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # Save any remaining recording
        if recording and rgb_frames:
            _save_recording(args.output, rgb_frames, depth_frames,
                            timestamps, depth_scale)


def _save_recording(output_path, rgb_frames, depth_frames, timestamps, depth_scale):
    """Save recorded frames to HDF5."""
    if not rgb_frames:
        return

    rgb = np.stack(rgb_frames, axis=0)  # (N, H, W, 3)
    depth = np.stack(depth_frames, axis=0)  # (N, H, W)
    ts = np.array(timestamps, dtype=np.float64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        ep = f.create_group("episode_000000")
        ep.create_dataset("sensors/camera/ego_rgb", data=rgb, compression="gzip",
                          chunks=(1, rgb.shape[1], rgb.shape[2], 3))
        ep.create_dataset("sensors/camera/ego_depth", data=depth, compression="gzip",
                          chunks=(1, depth.shape[1], depth.shape[2]))
        ep.create_dataset("timestamp", data=ts, compression="gzip")
        ep.attrs["num_frames"] = len(rgb)
        ep.attrs["depth_scale"] = depth_scale
        ep.attrs["resolution"] = f"{rgb.shape[2]}x{rgb.shape[1]}"

    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  Recording saved: {output_path} ({len(rgb)} frames, {size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

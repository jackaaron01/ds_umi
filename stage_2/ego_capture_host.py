#!/usr/bin/env python3
"""
HOST-side RealSense EGO capture script.

Run on the HOST (not Docker) to capture RealSense frames.
Saves snapshots to the shared volume for Docker to use.
Minimal dependency: just pyrealsense2 (pip install pyrealsense2).

Usage (on HOST):
    pip install pyrealsense2 opencv-python
    python3 stage_2/ego_capture_host.py              # view only
    python3 stage_2/ego_capture_host.py --save       # save snapshots
    python3 stage_2/ego_capture_host.py --record     # record to HDF5
"""
import sys, os, time, argparse
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("Please install: pip install pyrealsense2 opencv-python")
    sys.exit(1)

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "outputs", "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data")


def main():
    parser = argparse.ArgumentParser(description="Host-side RealSense EGO capture")
    parser.add_argument("--save", action="store_true", help="Save snapshots")
    parser.add_argument("--record", action="store_true", help="Record to HDF5")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    print("Starting RealSense...")
    pipeline.start(config)
    print("Camera ready. Controls: 's'=snapshot, 'r'=record, 'q'=quit")

    recording = args.record
    frames_rgb, frames_depth, timestamps = [], [], []
    frame_count = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            rgb = np.asanyarray(color.get_data())
            d = np.asanyarray(depth.get_data())
            depth_cmap = cv2.applyColorMap(
                cv2.convertScaleAbs(d, alpha=0.03), cv2.COLORMAP_JET)

            display = np.hstack((rgb, depth_cmap))
            status = "[REC]" if recording else "[LIVE]"
            cv2.putText(display, f"{status} Frames: {frame_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.imshow("RealSense EGO (RGB + Depth)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(os.path.join(OUTDIR, "ego_host_rgb.png"), rgb)
                cv2.imwrite(os.path.join(OUTDIR, "ego_host_depth.png"), depth_cmap)
                print(f"Snapshots saved to {OUTDIR}/")
                frame_count += 1
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print("Recording STARTED")
                    frames_rgb, frames_depth, timestamps = [], [], []
                else:
                    _save(args.output or os.path.join(DATA_DIR, "ego_real.h5"),
                          frames_rgb, frames_depth, timestamps)
                    frames_rgb, frames_depth, timestamps = [], [], []

            if recording:
                frames_rgb.append(rgb.copy())
                frames_depth.append(d.copy())
                timestamps.append(time.time())
                frame_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if recording and frames_rgb:
            _save(args.output or os.path.join(DATA_DIR, "ego_real.h5"),
                  frames_rgb, frames_depth, timestamps)


def _save(path, rgb_list, depth_list, ts_list):
    import h5py
    rgb = np.stack(rgb_list, 0)
    depth = np.stack(depth_list, 0)
    ts = np.array(ts_list, dtype=np.float64)
    with h5py.File(path, "w") as f:
        ep = f.create_group("episode_000000")
        ep.create_dataset("sensors/camera/ego_rgb", data=rgb,
                          compression="gzip", chunks=(1, *rgb.shape[1:]))
        ep.create_dataset("sensors/camera/ego_depth", data=depth,
                          compression="gzip", chunks=(1, *depth.shape[1:]))
        ep.create_dataset("timestamp", data=ts)
        ep.attrs["num_frames"] = len(rgb)
    print(f"Saved {len(rgb)} frames to {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data quality filters for UMI teleop recordings.

Each filter takes an HDF5 file path and returns a QualityReport with pass/fail
status and per-check details. Designed to be run as a pre-processing step before
adding episodes to a LeRobot dataset.

Usage:
    python3 quality_filters.py episode_000000.h5
    python3 quality_filters.py /path/to/umi_recordings/  # batch mode
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Thresholds ──────────────────────────────────────────────────────────────
MIN_EPISODE_STEPS = 30
MAX_TIMESTAMP_GAP = 0.100  # seconds, 3x the 30Hz expected interval
MAX_FRAME_DROP_RATIO = 0.1
MAX_JOINT_VELOCITY = 3.14  # rad/s
MAX_JOINT_DELTA = 0.3  # rad
VELOCITY_SPIKE_THRESH = 3.14  # rad/s — values above this are "spikes"
MAX_CONSECUTIVE_SPIKES = 3
GRIPPER_RANGE = (0.0, 1.0)
JOINT_LIMIT_MARGIN = 0.05  # rad from limit triggers warning
MIN_JOINT_VARIATION = 0.001  # rad — episode with less variation is "no motion"
ZERO_COMMAND_THRESHOLD = 0.9  # fraction of frames with no motion → low quality
MIN_EFFECTIVE_FPS = 10.0
MAX_CLOCK_DRIFT = 0.050  # seconds


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QualityReport:
    filepath: str
    num_steps: int
    effective_fps: float
    passed: bool = True
    checks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def analyze_episode(h5_path: str) -> QualityReport:
    """Run all quality checks on a single HDF5 episode file."""
    import h5py

    report = QualityReport(filepath=h5_path, num_steps=0, effective_fps=0.0)

    with h5py.File(h5_path, "r") as f:
        episodes = [k for k in f.keys() if k.startswith("episode_")]
        if not episodes:
            report.passed = False
            report.warnings.append("No episode group found in HDF5 file")
            return report
        ep = f[episodes[0]]

        # Load data
        if "joint_command/position" not in ep:
            report.passed = False
            report.warnings.append("Missing joint_command/position dataset")
            return report

        cmd = ep["joint_command/position"][:]
        report.num_steps = cmd.shape[0]
        has_ts = "joint_command/position_timestamp" in ep

        # ── 1. Completeness checks ──
        _check_min_steps(report)
        if has_ts:
            _check_timestamps(report, ep)
        _check_effective_fps(report, has_ts, ep, cmd)

        # ── 2. Kinematic checks ──
        _check_joint_velocity(report, cmd)
        _check_joint_delta(report, cmd)
        _check_velocity_spikes(report, cmd)

        # ── 3. Semantic checks ──
        _check_gripper_range(report, ep)
        _check_joint_limits(report, cmd)
        _check_no_motion(report, cmd)

    # Aggregate pass/fail
    report.passed = all(c.passed for c in report.checks)
    return report


def _check_min_steps(report: QualityReport):
    passed = report.num_steps >= MIN_EPISODE_STEPS
    report.checks.append(CheckResult(
        "min_steps", passed,
        f"{report.num_steps} steps (min: {MIN_EPISODE_STEPS})"
    ))


def _check_timestamps(report: QualityReport, ep):
    ts = ep["joint_command/position_timestamp"][:].flatten()
    intervals = np.diff(ts)
    gaps = np.sum(intervals > MAX_TIMESTAMP_GAP)
    drop_ratio = gaps / max(len(intervals), 1)
    passed = drop_ratio <= MAX_FRAME_DROP_RATIO
    report.checks.append(CheckResult(
        "timestamp_gaps", passed,
        f"{gaps} gaps > {MAX_TIMESTAMP_GAP*1000:.0f}ms ({drop_ratio:.1%})"
    ))


def _check_effective_fps(report: QualityReport, has_ts: bool, ep, cmd):
    if has_ts:
        ts = ep["joint_command/position_timestamp"][:].flatten()
        if len(ts) > 1:
            duration = ts[-1] - ts[0]
            fps = report.num_steps / max(duration, 1e-6)
            report.effective_fps = fps
            passed = fps >= MIN_EFFECTIVE_FPS
            report.checks.append(CheckResult(
                "effective_fps", passed,
                f"{fps:.1f} Hz (min: {MIN_EFFECTIVE_FPS})"
            ))
            return
    report.effective_fps = 0.0
    report.checks.append(CheckResult("effective_fps", False, "no timestamp data"))


def _check_joint_velocity(report: QualityReport, cmd: np.ndarray):
    if report.num_steps < 2:
        report.checks.append(CheckResult("joint_velocity", False, "too few steps"))
        return
    dt = 1.0 / max(report.effective_fps, MIN_EFFECTIVE_FPS)
    velocities = np.abs(np.diff(cmd, axis=0)) / dt
    max_vel = np.max(velocities)
    passed = max_vel <= MAX_JOINT_VELOCITY * 1.5  # 50% tolerance for transients
    report.checks.append(CheckResult(
        "joint_velocity", passed,
        f"max={max_vel:.2f} rad/s (limit: {MAX_JOINT_VELOCITY})"
    ))


def _check_joint_delta(report: QualityReport, cmd: np.ndarray):
    if report.num_steps < 2:
        return
    max_delta = np.max(np.abs(np.diff(cmd, axis=0)))
    passed = max_delta <= MAX_JOINT_DELTA
    report.checks.append(CheckResult(
        "joint_delta", passed,
        f"max_delta={max_delta:.3f} rad (limit: {MAX_JOINT_DELTA})"
    ))


def _check_velocity_spikes(report: QualityReport, cmd: np.ndarray):
    if report.num_steps < 2:
        return
    dt = 1.0 / max(report.effective_fps, MIN_EFFECTIVE_FPS)
    velocities = np.abs(np.diff(cmd, axis=0)) / dt
    # Find consecutive spikes
    spike_mask = np.any(velocities > VELOCITY_SPIKE_THRESH, axis=1)
    max_consecutive = 0
    current = 0
    for s in spike_mask:
        if s:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0
    passed = max_consecutive <= MAX_CONSECUTIVE_SPIKES
    total_spikes = int(np.sum(spike_mask))
    report.checks.append(CheckResult(
        "velocity_spikes", passed,
        f"{total_spikes} spikes, max {max_consecutive} consecutive (limit: {MAX_CONSECUTIVE_SPIKES})"
    ))


def _check_gripper_range(report: QualityReport, ep):
    if "gripper/command" not in ep:
        return
    gcmd = ep["gripper/command"][:]
    out_of_range = int(np.sum((gcmd < GRIPPER_RANGE[0]) | (gcmd > GRIPPER_RANGE[1])))
    passed = out_of_range == 0
    report.checks.append(CheckResult(
        "gripper_range", passed,
        f"{out_of_range} values outside {GRIPPER_RANGE}"
    ))


def _check_joint_limits(report: QualityReport, cmd: np.ndarray):
    from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS
    limits = np.array(XARM6_JOINT_LIMITS)
    near_limit = 0
    for j in range(cmd.shape[1]):
        dist_lower = cmd[:, j] - (limits[j, 0] + JOINT_LIMIT_MARGIN)
        dist_upper = (limits[j, 1] - JOINT_LIMIT_MARGIN) - cmd[:, j]
        near_limit += int(np.sum(dist_lower < 0)) + int(np.sum(dist_upper < 0))
    passed = near_limit == 0
    report.checks.append(CheckResult(
        "joint_limits", passed,
        f"{near_limit} frames near limit (margin={JOINT_LIMIT_MARGIN} rad)"
    ))


def _check_no_motion(report: QualityReport, cmd: np.ndarray):
    total_variation = np.sum(np.std(cmd, axis=0))
    passed = total_variation >= MIN_JOINT_VARIATION
    report.checks.append(CheckResult(
        "no_motion", passed,
        f"total variation={total_variation:.4f} rad (min: {MIN_JOINT_VARIATION})"
    ))


def batch_check(directory: str) -> list:
    """Run quality checks on all HDF5 files in a directory."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".h5"):
            path = os.path.join(directory, fname)
            results.append(analyze_episode(path))
    return results


def print_report(report: QualityReport):
    """Pretty-print a single quality report."""
    status = "PASS" if report.passed else "FAIL"
    print(f"\n{'='*70}")
    print(f"  {os.path.basename(report.filepath)}  [{status}]")
    print(f"  Steps: {report.num_steps}  |  Effective FPS: {report.effective_fps:.1f}")
    print(f"{'='*70}")
    for c in report.checks:
        flag = "+" if c.passed else "!"
        print(f"  [{flag}] {c.name}: {c.detail}")
    for w in report.warnings:
        print(f"  [*] {w}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 quality_filters.py <episode.h5|directory/>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        results = batch_check(target)
        passed = sum(1 for r in results if r.passed)
        for r in results:
            print_report(r)
        print(f"\nSummary: {passed}/{len(results)} episodes passed quality checks")
    else:
        report = analyze_episode(target)
        print_report(report)
        sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()

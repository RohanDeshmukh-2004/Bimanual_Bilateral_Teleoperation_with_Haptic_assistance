#!/usr/bin/env python3
"""
teleop_bimanual.py — Bilateral bimanual teleoperation with Haptic Feedback,
                     HDF5 episode recording, Synchronized Camera Capture,
                     and PID-based Failure-Aware Corrective Response.
"""

import os
import sys
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import subprocess
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from constants import (
    DT,
    LEADER_START_POSE,   FOLLOWER_START_POSE,
    LEADER_RIGHT_GRIPPER_OPEN, FOLLOWER_RIGHT_GRIPPER_OPEN,
    LEADER_RIGHT_GRIPPER_CLOSE,
    LEADER_RIGHT,  LEADER_LEFT,
    FOLLOWER_RIGHT, FOLLOWER_LEFT,
    LEADER_JOINT_NAMES,
    FOLLOWER_JOINT_NAMES,
    FOLLOWER_NUM_JOINTS,
    LEADER_TO_FOLLOWER_MAP,
    FOLLOWER_RIGHT_GRIPPER_CLOSE,
    LEADER_LEFT_GRIPPER_OPEN, LEADER_LEFT_GRIPPER_CLOSE,   # Updated Left
    FOLLOWER_LEFT_GRIPPER_OPEN, FOLLOWER_LEFT_GRIPPER_CLOSE,
)
from robot_utils import ArmController


# ── HAPTIC FEEDBACK PARAMETERS (FINE-TUNED) ──────────────────────────────────
HAPTIC_SCALE         = 0.00004  # Eased scaling: requires much less human effort
HAPTIC_MAX_OFFSET    = 0.02     # Lower ceiling to prevent current overshoot/jerks
HAPTIC_SPRING_GAIN   = 0.08     # Softens the virtual spring pushback

# Heavier filtering to completely eliminate motor chatter/noise:
HAPTIC_EFFORT_ALPHA  = 0.92     # Heavy low-pass on raw current to kill noise
HAPTIC_OUTPUT_ALPHA  = 0.85     # Smoother position command trajectory
HAPTIC_RELEASE_ALPHA = 0.50     # Soft, non-aggressive snap-back on release

HAPTIC_DEADBANDS = {
    "waist":        200.0,      # Widened deadbands ensure free-space movement
    "shoulder":     200.0,      # is completely effortless and friction-free
    "elbow":        350.0,
    "wrist_angle":  150.0,
    "wrist_rotate": 150.0,
}
# ─────────────────────────────────────────────────────────────────────────────

# ── HDF5 & VISION RECORDING PARAMETERS ───────────────────────────────────────
DATASET_DIR = os.path.expanduser("~/aloha_data/episodes")
T_MAX       = 2000

# Confirmed hardware camera device nodes (/dev/videoX)
CAMERA_CONFIG = {
    "cam_top":  "/dev/video0",
    "cam_front": "/dev/video2",
}
IMG_WIDTH  = 640
IMG_HEIGHT = 480
# ─────────────────────────────────────────────────────────────────────────────

# ── GRIPPER THERMAL PROTECTION PARAMETERS ────────────────────────────────────
# Update the follower gripper command at 10 Hz instead of 50 Hz.
# Sending a stalled position command at 50 Hz causes the vx300s gripper motor
# (XM430) to continuously re-assert full current against the load, overheating
# and triggering Dynamixel thermal shutdown in 2-4 minutes.
# At 10 Hz the motor re-asserts only once every 5 steps — 5x less sustained
# current draw on a stalled gripper — while remaining imperceptible to the operator.
#
# NOTE: The per-command deadband that previously accompanied this rate-limiter
# has been intentionally removed. The deadband suppressed re-open commands when
# the operator opens the leader gripper slowly after a handover (delta < deadband
# per 5-step window), causing the follower gripper to freeze in the closed
# position indefinitely. The 10 Hz rate limiter alone provides sufficient thermal
# protection — every command that arrives at the tick boundary is now transmitted
# unconditionally, keeping the gripper responsive throughout teleoperation.
GRIPPER_UPDATE_EVERY_N_STEPS = 5   # 50 Hz / 5 = 10 Hz effective gripper rate
# ─────────────────────────────────────────────────────────────────────────────

# ── PID CORRECTIVE RESPONSE PARAMETERS ───────────────────────────────────────
# Per-joint tracking error thresholds (radians) that trigger a failure flag.
# Tuned conservatively for vx300s — widen if you get too many false positives.
PID_FAILURE_THRESHOLDS = np.array([
    0.08,   # joint 0 — waist       (large joint, loose threshold)
    0.06,   # joint 1 — shoulder    (high load, tighter)
    0.06,   # joint 2 — elbow
    0.04,   # joint 3 — wrist_angle (precision joint)
    0.04,   # joint 4 — wrist_rotate
])

# PID gains (joint-space, radians). Start conservative — increase Kp gradually.
PID_KP = np.array([0.40, 0.50, 0.50, 0.60, 0.60])
PID_KI = np.array([0.01, 0.01, 0.01, 0.02, 0.02])
PID_KD = np.array([0.05, 0.05, 0.05, 0.08, 0.08])

# How strongly PID correction blends into the operator command.
# 0.0 = pure operator, 1.0 = full PID override. Keep ≤ 0.5 during initial tests.
PID_BLEND_ALPHA = 0.35

# Noise floor: errors below this (radians) are ignored — prevents correction
# chatter on healthy joints that are within normal servo jitter.
PID_ACTIVATION_THRESHOLD = 0.02

# Hard cap on correction magnitude per joint (radians).
# Guards against the PID fighting a torque-saturated Dynamixel servo.
PID_MAX_CORRECTION = 0.05

# Effort level (raw Dynamixel units) above which a joint is considered
# torque-saturated — PID correction is zeroed on saturated joints to avoid
# making thermal overload worse.
PID_EFFORT_SATURATION = 800.0

# Integral windup clamp (radians). Prevents runaway accumulation.
PID_INTEGRAL_CLAMP = 0.10
# ─────────────────────────────────────────────────────────────────────────────


class TrackingPIDController:
    """
    Per-joint PID controller that measures follower tracking error and generates
    a corrective offset blended on top of the operator's joint-space command.

    Tracking error is defined as:
        e[t] = cmd_sent[t-1] - actual_achieved[t]

    This is NOT a servo PID — it is a failure-correction overlay. The Dynamixel
    servos already have their own internal position PID. This controller operates
    one level above: it detects when the follower arm as a whole has drifted from
    the commanded trajectory and injects a soft corrective nudge into the next
    command, blended with the operator's intent.

    Failure detection:
        A joint is flagged as "in failure" when |e[t]| > PID_FAILURE_THRESHOLDS[j].
        The flag is recorded in HDF5 so failure-timestep analysis is possible
        after collection.

    Saturation guard:
        If follower effort on a joint exceeds PID_EFFORT_SATURATION, correction
        is zeroed for that joint — pushing a torque-saturated motor further into
        its limit worsens heating without improving tracking.
    """

    def __init__(self, n_joints: int, dt: float):
        self.n   = n_joints
        self.dt  = dt

        self._integral   = np.zeros(n_joints, dtype=np.float64)
        self._prev_error = np.zeros(n_joints, dtype=np.float64)

        # Expose for external read-out (recording, logging)
        self.tracking_error  = np.zeros(n_joints, dtype=np.float64)
        self.failure_flags   = np.zeros(n_joints, dtype=bool)
        self.any_failure     = False

    def reset(self):
        """Reset integrators. Call when the operator takes over after a failure."""
        self._integral[:]   = 0.0
        self._prev_error[:] = 0.0

    def step(
        self,
        cmd_prev:      np.ndarray,   # command sent to follower at t-1  (n_joints,)
        actual_now:    np.ndarray,   # follower actual qpos at t         (n_joints,)
        operator_cmd:  np.ndarray,   # raw operator command for t        (n_joints,)
        follower_eff:  np.ndarray,   # follower effort at t              (n_joints,)
    ) -> np.ndarray:
        """
        Compute and return the corrected follower command for this timestep.

        Side effects (readable after call):
            self.tracking_error  — raw per-joint error (radians)
            self.failure_flags   — per-joint bool, True if above threshold
            self.any_failure     — True if any joint is in failure
        """
        # ── 1. Tracking error ────────────────────────────────────────────────
        error = cmd_prev - actual_now          # shape: (n_joints,)
        self.tracking_error = error.copy()

        # ── 2. Failure detection ─────────────────────────────────────────────
        self.failure_flags = np.abs(error) > PID_FAILURE_THRESHOLDS[:self.n]
        self.any_failure   = bool(self.failure_flags.any())

        # ── 3. Saturation guard — zero correction on overloaded joints ───────
        saturated = np.abs(follower_eff[:self.n]) > PID_EFFORT_SATURATION

        # ── 4. Noise floor gate — only correct joints above activation thr. ──
        active = (np.abs(error) > PID_ACTIVATION_THRESHOLD) & (~saturated)

        # ── 5. PID computation ───────────────────────────────────────────────
        self._integral  += error * self.dt
        self._integral   = np.clip(self._integral,
                                   -PID_INTEGRAL_CLAMP, PID_INTEGRAL_CLAMP)

        derivative       = (error - self._prev_error) / self.dt
        self._prev_error = error.copy()

        correction = (
            PID_KP[:self.n] * error
            + PID_KI[:self.n] * self._integral
            + PID_KD[:self.n] * derivative
        ) * active   # zero on inactive / saturated joints

        # ── 6. Hard clamp ────────────────────────────────────────────────────
        correction = np.clip(correction, -PID_MAX_CORRECTION, PID_MAX_CORRECTION)

        # ── 7. Blend with operator command ───────────────────────────────────
        corrected_cmd = operator_cmd + PID_BLEND_ALPHA * correction

        return corrected_cmd


class CameraStream:
    """Dedicated background grabber thread optimized for uncompressed YUYV streams."""

    def __init__(self, name: str, device_path: str, width: int, height: int):
        self.name = name

        # 0. PRE-EMPTIVE CLEANUP: Ensure no zombie process has locked the device
        try:
            subprocess.run(["fuser", "-k", device_path], stderr=subprocess.DEVNULL)
            time.sleep(0.5) # Give the OS time to release the file handle
        except Exception:
            pass

        # 1. Force the V4L2 backend explicitly
        self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

        # 2. Force YUYV format BEFORE requesting resolution to prevent USB bus overflow
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, int(1.0 / DT))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            print(f"[WARN] Camera '{name}' ({device_path}) failed to open. Check USB bandwidth.")

        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, img = self.cap.read()
            if ret and img is not None:
                with self.lock:
                    self.frame = img
            else:
                time.sleep(0.01)

    def read(self) -> np.ndarray:
        with self.lock:
            return self.frame.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

class ContinuousGripper:
    """Proportional gripper tracking to prevent follower motor thermal overload."""

    def __init__(
        self,
        l_open: float,
        l_close: float,
        f_open: float,
        f_close: float,
    ):
        self.l_open = l_open
        self.l_close = l_close
        self.f_open = f_open
        self.f_close = f_close

    def get_cmd(self, leader_raw: float) -> float:
        # Linearly map the leader's actual position to the follower's range
        ratio = (leader_raw - self.l_open) / (self.l_close - self.l_open + 1e-6)

        # Clamp the ratio cleanly between 0.0 and 1.0
        ratio = max(0.0, min(1.0, ratio))

        return self.f_open + ratio * (self.f_close - self.f_open)


class EpisodeRecorder:
    """Pre-allocated ring-buffer for kinematics and multi-camera streams."""

    def __init__(self, t_max: int, dataset_dir: str, cam_names: list[str]):
        self.t_max = t_max
        self.dataset_dir = dataset_dir
        self.cam_names = cam_names
        self.step = 0

        self.buf_qpos = np.zeros((t_max, 26), dtype=np.float32)
        self.buf_action = np.zeros((t_max, 14), dtype=np.float32)
        self.buf_effort_r = np.zeros((t_max, 6), dtype=np.float32)
        self.buf_effort_l = np.zeros((t_max, 6), dtype=np.float32)
        self.buf_haptic_off_r = np.zeros((t_max, 5), dtype=np.float32)
        self.buf_haptic_off_l = np.zeros((t_max, 5), dtype=np.float32)
        self.buf_haptic_active = np.zeros((t_max, 2), dtype=np.bool_)

        # ── PID failure-aware additions ───────────────────────────────────────
        self.buf_tracking_err_r  = np.zeros((t_max, 5), dtype=np.float32)
        self.buf_tracking_err_l  = np.zeros((t_max, 5), dtype=np.float32)
        self.buf_failure_r       = np.zeros(t_max, dtype=np.bool_)
        self.buf_failure_l       = np.zeros(t_max, dtype=np.bool_)
        # ─────────────────────────────────────────────────────────────────────

        self.buf_images = {
            cam: np.zeros((t_max, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            for cam in cam_names
        }

    def record(
        self,
        lr_arm,
        lr_grip,
        ll_arm,
        ll_grip,
        fr_arm_act,
        fr_grip_act,
        fl_arm_act,
        fl_grip_act,
        fr_arm_cmd,
        fr_grip_cmd,
        fl_arm_cmd,
        fl_grip_cmd,
        effort_r_filt,
        effort_l_filt,
        haptic_off_r,
        haptic_off_l,
        any_haptic_r,
        any_haptic_l,
        cam_frames: dict,
        # ── PID additions (keyword args with defaults for backward compat) ───
        tracking_err_r: np.ndarray = None,
        tracking_err_l: np.ndarray = None,
        any_failure_r:  bool = False,
        any_failure_l:  bool = False,
    ):
        if self.step >= self.t_max:
            return

        k = self.step

        self.buf_qpos[k] = np.concatenate([
            lr_arm, [lr_grip], ll_arm, [ll_grip],
            fr_arm_act, [fr_grip_act], fl_arm_act, [fl_grip_act]
        ])

        self.buf_action[k] = np.concatenate([
            fr_arm_cmd, [fr_grip_cmd], fl_arm_cmd, [fl_grip_cmd]
        ])

        self.buf_effort_r[k] = effort_r_filt.astype(np.float32)
        self.buf_effort_l[k] = effort_l_filt.astype(np.float32)
        self.buf_haptic_off_r[k] = haptic_off_r.astype(np.float32)
        self.buf_haptic_off_l[k] = haptic_off_l.astype(np.float32)
        self.buf_haptic_active[k] = [any_haptic_r, any_haptic_l]

        # ── PID additions ─────────────────────────────────────────────────────
        if tracking_err_r is not None:
            self.buf_tracking_err_r[k] = tracking_err_r[:5].astype(np.float32)
        if tracking_err_l is not None:
            self.buf_tracking_err_l[k] = tracking_err_l[:5].astype(np.float32)
        self.buf_failure_r[k] = any_failure_r
        self.buf_failure_l[k] = any_failure_l
        # ─────────────────────────────────────────────────────────────────────

        for cam_name, frame in cam_frames.items():
            if cam_name in self.buf_images:
                self.buf_images[cam_name][k] = frame

        self.step += 1

    def _next_episode_index(self) -> int:
        os.makedirs(self.dataset_dir, exist_ok=True)
        existing = [
            f
            for f in os.listdir(self.dataset_dir)
            if f.startswith("episode_") and f.endswith(".hdf5")
        ]
        if not existing:
            return 0
        indices = []
        for fname in existing:
            try:
                indices.append(
                    int(fname.replace("episode_", "").replace(".hdf5", ""))
                )
            except ValueError:
                pass
        return max(indices) + 1 if indices else 0

    def save(self) -> str:
        T = self.step
        if T == 0:
            print("[Recorder] No data recorded — skipping save.")
            return ""

        ep_idx = self._next_episode_index()
        fpath = os.path.join(self.dataset_dir, f"episode_{ep_idx:04d}.hdf5")

        print("\n[Recorder] Serializing tensors to HDF5 format...")
        with h5py.File(fpath, "w", rdcc_nbytes=1024**2 * 10) as f:
            f.attrs["compress_len"] = T
            f.attrs["sim"] = False

            obs = f.create_group("observations")
            obs.create_dataset(
                "qpos",
                data=self.buf_qpos[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            obs.create_dataset(
                "effort_right",
                data=self.buf_effort_r[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            obs.create_dataset(
                "effort_left",
                data=self.buf_effort_l[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            obs.create_dataset(
                "haptic_offset_right",
                data=self.buf_haptic_off_r[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            obs.create_dataset(
                "haptic_offset_left",
                data=self.buf_haptic_off_l[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )

            # ── PID additions ─────────────────────────────────────────────────
            obs.create_dataset(
                "tracking_error_right",
                data=self.buf_tracking_err_r[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            obs.create_dataset(
                "tracking_error_left",
                data=self.buf_tracking_err_l[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            # ─────────────────────────────────────────────────────────────────

            images_group = obs.create_group("images")
            for cam_name in self.cam_names:
                print(f"  --> Compressing stream: {cam_name}")
                images_group.create_dataset(
                    cam_name,
                    data=self.buf_images[cam_name][:T],
                    dtype="uint8",
                    chunks=(1, IMG_HEIGHT, IMG_WIDTH, 3),
                    compression="gzip",
                    compression_opts=2,
                )

            f.create_dataset(
                "action",
                data=self.buf_action[:T],
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "haptic_active",
                data=self.buf_haptic_active[:T],
                dtype=np.bool_,
            )

            # ── PID additions ─────────────────────────────────────────────────
            f.create_dataset(
                "failure_right",
                data=self.buf_failure_r[:T],
                dtype=np.bool_,
            )
            f.create_dataset(
                "failure_left",
                data=self.buf_failure_l[:T],
                dtype=np.bool_,
            )
            # ─────────────────────────────────────────────────────────────────

        haptic_fraction = self.buf_haptic_active[:T].any(axis=1).mean()

        # ── PID additions — summary stats at save time ────────────────────────
        failure_fraction_r = self.buf_failure_r[:T].mean()
        failure_fraction_l = self.buf_failure_l[:T].mean()
        mean_err_r = np.abs(self.buf_tracking_err_r[:T]).mean(axis=0)
        mean_err_l = np.abs(self.buf_tracking_err_l[:T]).mean(axis=0)
        # ─────────────────────────────────────────────────────────────────────

        print(f"[Recorder] Complete! Saved {T} steps → {fpath}")
        print(f"           Haptic active fraction:      {haptic_fraction:.2%}")
        print(f"           Failure fraction R / L:      {failure_fraction_r:.2%} / {failure_fraction_l:.2%}")
        print(f"           Mean |tracking error| R (rad): {np.round(mean_err_r, 4)}")
        print(f"           Mean |tracking error| L (rad): {np.round(mean_err_l, 4)}")

        # ── Episode quality gate — warn if failure rate is too high ───────────
        if failure_fraction_r > 0.15 or failure_fraction_l > 0.15:
            print(
                f"[Recorder] WARNING: failure fraction exceeded 15% on one or both arms. "
                f"Consider discarding this episode — the operator could not maintain "
                f"tracking for a large fraction of steps."
            )
        # ─────────────────────────────────────────────────────────────────────

        return fpath


def make_follower_cmd(leader_qpos: np.ndarray) -> np.ndarray:
    cmd = np.zeros(FOLLOWER_NUM_JOINTS)
    for l_idx, f_idx in LEADER_TO_FOLLOWER_MAP:
        cmd[f_idx] = leader_qpos[l_idx]
    return cmd


def wait_for_arms_ready(*controllers: ArmController, timeout: float = 20.0):
    print("Waiting for all arms to publish joint states …")
    t0 = time.time()
    while not all(c.is_ready() for c in controllers):
        if time.time() - t0 > timeout:
            raise RuntimeError("Timed out waiting for joint states.")
        time.sleep(0.1)
    print("All arms are ready ✓")


def setup_leaders(*leaders: ArmController):
    for leader in leaders:
        leader.set_operating_mode(
            "group",
            "arm",
            "position",
            profile_velocity=40,
            profile_acceleration=15,
        )
        leader.torque_enable("group", "arm", False)
        leader.set_operating_mode(
            "single",
            "gripper",
            "position",
            profile_velocity=131,
            profile_acceleration=15,
        )
        leader.torque_enable("single", "gripper", False)


def setup_followers(*followers: ArmController):
    for follower in followers:
        follower.set_operating_mode(
            "group",
            "arm",
            "position",
            profile_velocity=131,
            profile_acceleration=15,
        )
        follower.set_operating_mode(
            "single",
            "gripper",
            "current based position",
            profile_velocity=131,
            profile_acceleration=15,
        )
        follower.torque_enable("group", "arm", True)
        follower.torque_enable("single", "gripper", True)


def countdown(seconds: int = 3):
    print("\n" + "=" * 60)
    print("READY.  Teleoperation starts in:")
    for i in range(seconds, 0, -1):
        print(f"  {i}…")
        time.sleep(1.0)
    print("GO — move the leader arms freely.")
    print("=" * 60 + "\n")


def compute_haptic_offset(
    joint_name: str,
    raw_effort: float,
    effort_state: float,
    prev_smooth: float,
) -> tuple[float, float, float]:
    joint_deadband = HAPTIC_DEADBANDS.get(joint_name, 150.0)

    effort_state = (
        HAPTIC_EFFORT_ALPHA * effort_state
        + (1.0 - HAPTIC_EFFORT_ALPHA) * raw_effort
    )

    if abs(effort_state) < joint_deadband:
        raw_offset = 0.0
    else:
        sign = 1.0 if effort_state > 0.0 else -1.0
        raw_offset = (
            -sign
            * (abs(effort_state) - joint_deadband)
            * HAPTIC_SCALE
            * HAPTIC_SPRING_GAIN
        )

    clamped_offset = max(-HAPTIC_MAX_OFFSET, min(HAPTIC_MAX_OFFSET, raw_offset))

    current_alpha = (
        HAPTIC_OUTPUT_ALPHA
        if abs(clamped_offset) > abs(prev_smooth)
        else HAPTIC_RELEASE_ALPHA
    )
    smooth_offset = (
        current_alpha * prev_smooth + (1.0 - current_alpha) * clamped_offset
    )

    return smooth_offset, effort_state, smooth_offset


def teleop_loop(
    leader_right,
    leader_left,
    follower_right,
    follower_left,
    gripper_right,
    gripper_left,
    recorder,
    cameras,
):
    print("Teleoperation active.\n")
    step = 0
    time_data, target_data, actual_data = [], [], []
    t0_loop = time.time()

    effort_state_r = np.zeros(FOLLOWER_NUM_JOINTS, dtype=np.float64)
    effort_state_l = np.zeros(FOLLOWER_NUM_JOINTS, dtype=np.float64)
    haptic_smooth_r = np.zeros(5, dtype=np.float64)
    haptic_smooth_l = np.zeros(5, dtype=np.float64)

    torque_on_r = False
    torque_on_l = False

    # ── PID controllers — one per arm, arm joints only (5 DOF, no gripper) ───
    # The gripper is already handled proportionally by ContinuousGripper.
    pid_right = TrackingPIDController(n_joints=5, dt=DT)
    pid_left  = TrackingPIDController(n_joints=5, dt=DT)

    # Initialise prev_cmd to the follower's actual start pose so the first
    # error measurement is ~0 rather than a spurious large value.
    prev_fr_cmd = np.zeros(5, dtype=np.float64)
    prev_fl_cmd = np.zeros(5, dtype=np.float64)
    _first_step = True          # flag to seed prev_cmd cleanly on step 0
    # ─────────────────────────────────────────────────────────────────────────

    try:
        while True:
            t_start = time.time()

            lr_arm = leader_right.get_arm_qpos()
            ll_arm = leader_left.get_arm_qpos()
            lr_grip = leader_right.get_gripper_qpos()
            ll_grip = leader_left.get_gripper_qpos()

            fr_eff = follower_right.get_arm_effort()
            fl_eff = follower_left.get_arm_effort()
            fr_arm_act = follower_right.get_arm_qpos()
            fl_arm_act = follower_left.get_arm_qpos()
            fr_grip_act = follower_right.get_gripper_qpos()
            fl_grip_act = follower_left.get_gripper_qpos()

            cam_frames = {name: cam.read() for name, cam in cameras.items()}

            # ── Seed prev_cmd on the very first step ─────────────────────────
            if _first_step:
                prev_fr_cmd = fr_arm_act[:5].copy()
                prev_fl_cmd = fl_arm_act[:5].copy()
                _first_step = False
            # ─────────────────────────────────────────────────────────────────

            # ── Operator intent (unchanged from original) ─────────────────────
            fr_cmd_operator = make_follower_cmd(lr_arm)
            fl_cmd_operator = make_follower_cmd(ll_arm)

            # Grippers now seamlessly track the analog leader position
            fr_grip_cmd = gripper_right.get_cmd(float(lr_grip))
            fl_grip_cmd = gripper_left.get_cmd(float(ll_grip))
            # ─────────────────────────────────────────────────────────────────

            # ── PID corrective step ───────────────────────────────────────────
            # Arms only (5 joints). Gripper is not PID-corrected.
            fr_cmd_corrected = pid_right.step(
                cmd_prev=prev_fr_cmd,
                actual_now=fr_arm_act[:5],
                operator_cmd=fr_cmd_operator[:5],
                follower_eff=fr_eff[:5],
            )
            fl_cmd_corrected = pid_left.step(
                cmd_prev=prev_fl_cmd,
                actual_now=fl_arm_act[:5],
                operator_cmd=fl_cmd_operator[:5],
                follower_eff=fl_eff[:5],
            )

            # Rebuild full FOLLOWER_NUM_JOINTS command vector — PID covers
            # joints 0-4; remaining joint(s) come from the operator map as-is.
            fr_cmd = fr_cmd_operator.copy()
            fl_cmd = fl_cmd_operator.copy()
            fr_cmd[:5] = fr_cmd_corrected
            fl_cmd[:5] = fl_cmd_corrected

            # Advance prev_cmd for next iteration
            prev_fr_cmd = fr_cmd[:5].copy()
            prev_fl_cmd = fl_cmd[:5].copy()
            # ─────────────────────────────────────────────────────────────────

            any_haptic_r = False
            haptic_cmd_r, haptic_off_r = np.zeros(5), np.zeros(5)

            for l_idx, f_idx in LEADER_TO_FOLLOWER_MAP:
                off, effort_state_r[f_idx], haptic_smooth_r[l_idx] = (
                    compute_haptic_offset(
                        LEADER_JOINT_NAMES[l_idx],
                        fr_eff[f_idx],
                        effort_state_r[f_idx],
                        haptic_smooth_r[l_idx],
                    )
                )
                haptic_off_r[l_idx] = off
                haptic_cmd_r[l_idx] = lr_arm[l_idx] + off
                if abs(off) > 0.001:
                    any_haptic_r = True

            if any_haptic_r:
                if not torque_on_r:
                    leader_right.torque_enable("group", "arm", True)
                    torque_on_r = True
                leader_right.set_arm_positions(haptic_cmd_r)
            elif torque_on_r:
                leader_right.torque_enable("group", "arm", False)
                torque_on_r = False

            any_haptic_l = False
            haptic_cmd_l, haptic_off_l = np.zeros(5), np.zeros(5)

            for l_idx, f_idx in LEADER_TO_FOLLOWER_MAP:
                off, effort_state_l[f_idx], haptic_smooth_l[l_idx] = (
                    compute_haptic_offset(
                        LEADER_JOINT_NAMES[l_idx],
                        fl_eff[f_idx],
                        effort_state_l[f_idx],
                        haptic_smooth_l[l_idx],
                    )
                )
                haptic_off_l[l_idx] = off
                haptic_cmd_l[l_idx] = ll_arm[l_idx] + off
                if abs(off) > 0.001:
                    any_haptic_l = True

            if any_haptic_l:
                if not torque_on_l:
                    leader_left.torque_enable("group", "arm", True)
                    torque_on_l = True
                leader_left.set_arm_positions(haptic_cmd_l)
            elif torque_on_l:
                leader_left.torque_enable("group", "arm", False)
                torque_on_l = False

            follower_right.set_arm_positions(fr_cmd)
            follower_left.set_arm_positions(fl_cmd)

            # ── Gripper thermal protection (rate-limited only) ────────────────
            # Commands are sent unconditionally on every 10 Hz tick.
            #
            # The previous deadband gate (only send if |cmd - prev_sent| > 0.002)
            # has been removed. It caused the follower gripper to freeze after
            # object handover: when the operator slowly re-opens the leader
            # gripper the per-tick delta stays below the deadband threshold, so
            # no re-open command was ever transmitted.
            #
            # The 10 Hz rate limiter alone (GRIPPER_UPDATE_EVERY_N_STEPS = 5)
            # provides sufficient thermal protection by reducing the stall
            # re-assertion rate by 5x. The HDF5-recorded fr_grip_cmd /
            # fl_grip_cmd values remain at full 50 Hz operator resolution.
            if step % GRIPPER_UPDATE_EVERY_N_STEPS == 0:
                follower_right.set_gripper_position(fr_grip_cmd)
                follower_left.set_gripper_position(fl_grip_cmd)
            # ─────────────────────────────────────────────────────────────────

            recorder.record(
                lr_arm,
                float(lr_grip),
                ll_arm,
                float(ll_grip),
                fr_arm_act,
                float(fr_grip_act),
                fl_arm_act,
                float(fl_grip_act),
                fr_cmd,
                float(fr_grip_cmd),
                fl_cmd,
                float(fl_grip_cmd),
                effort_state_r.copy(),
                effort_state_l.copy(),
                haptic_off_r,
                haptic_off_l,
                any_haptic_r,
                any_haptic_l,
                cam_frames,
                # ── PID additions ────────────────────────────────────────────
                tracking_err_r=pid_right.tracking_error,
                tracking_err_l=pid_left.tracking_error,
                any_failure_r=pid_right.any_failure,
                any_failure_l=pid_left.any_failure,
                # ────────────────────────────────────────────────────────────
            )

            time_data.append(t_start - t0_loop)
            target_data.append(fr_cmd.copy())
            actual_data.append(fr_arm_act.copy())

            step += 1
            if step % 100 == 0:
                print(
                    f"  step {step:6d}  | Haptic R: {any_haptic_r} | "
                    f"Haptic L: {any_haptic_l} | Recorded: {recorder.step} | "
                    # ── PID additions ────────────────────────────────────────
                    f"Failure R: {pid_right.any_failure} | "
                    f"Failure L: {pid_left.any_failure} | "
                    f"MaxErr R: {np.abs(pid_right.tracking_error).max():.3f} rad"
                    # ────────────────────────────────────────────────────────
                )

            remainder = DT - (time.time() - t_start)
            if remainder > 0:
                time.sleep(remainder)

    except KeyboardInterrupt:
        print("\nTeleoperation stopped. Halting vision threads...")
        for cam in cameras.values():
            cam.stop()
        recorder.save()
        _plot_tracking(time_data, target_data, actual_data)


def _plot_tracking(time_data, target_data, actual_data):
    if not time_data:
        return
    t_np = np.array(time_data)
    tgt_np = np.array(target_data)
    act_np = np.array(actual_data)

    joint_names = ["Waist", "Shoulder", "Elbow"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Teleoperation Tracking Performance — Right Arm", fontsize=14
    )

    for i, jname in enumerate(joint_names):
        axs[i].plot(
            t_np, tgt_np[:, i], label="Leader Command (Target)", linewidth=2
        )
        axs[i].plot(
            t_np,
            act_np[:, i],
            label="Follower Actual",
            linestyle="--",
            linewidth=2,
        )
        axs[i].set_ylabel(f"{jname} (rad)")
        axs[i].legend(loc="upper right")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig("tracking_performance.png", dpi=150)
    plt.show()


def main():
    rclpy.init(args=sys.argv)
    node = Node("aloha_teleop")

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    active_cams = list(CAMERA_CONFIG.keys())
    recorder = EpisodeRecorder(
        t_max=T_MAX, dataset_dir=DATASET_DIR, cam_names=active_cams
    )

    print("[Vision] Initializing multi-camera hardware capture threads...")
    cameras = {
        name: CameraStream(name, idx, IMG_WIDTH, IMG_HEIGHT)
        for name, idx in CAMERA_CONFIG.items()
    }

    print(f"[Recorder] Episodes will be saved to: {DATASET_DIR}")
    print(
        f"[Recorder] Max episode length: {T_MAX} steps ({T_MAX * DT:.0f} s at {1/DT:.0f} Hz)"
    )

    try:
        leader_right = ArmController(node, LEADER_RIGHT, LEADER_JOINT_NAMES)
        leader_left = ArmController(node, LEADER_LEFT, LEADER_JOINT_NAMES)
        follower_right = ArmController(
            node, FOLLOWER_RIGHT, FOLLOWER_JOINT_NAMES
        )
        follower_left = ArmController(node, FOLLOWER_LEFT, FOLLOWER_JOINT_NAMES)

        wait_for_arms_ready(
            leader_right, leader_left, follower_right, follower_left
        )

        gripper_right = ContinuousGripper(
            LEADER_RIGHT_GRIPPER_OPEN,
            LEADER_RIGHT_GRIPPER_CLOSE,
            FOLLOWER_RIGHT_GRIPPER_OPEN,
            FOLLOWER_RIGHT_GRIPPER_CLOSE,       
        )
       
        gripper_left = ContinuousGripper(
            LEADER_LEFT_GRIPPER_OPEN,
            LEADER_LEFT_GRIPPER_CLOSE,
            FOLLOWER_LEFT_GRIPPER_OPEN,
            FOLLOWER_LEFT_GRIPPER_CLOSE,
        )

        print("Configuring follower operating modes …")
        setup_followers(follower_right, follower_left)

        print("Configuring leader operating modes …")
        setup_leaders(leader_right, leader_left)

        print("Moving followers to home pose …")
        follower_right.go_to_home_pose(
            FOLLOWER_START_POSE, FOLLOWER_RIGHT_GRIPPER_OPEN
        )
        follower_left.go_to_home_pose(
            FOLLOWER_START_POSE, FOLLOWER_LEFT_GRIPPER_OPEN
        )

        countdown(seconds=3)

        teleop_loop(
            leader_right,
            leader_left,
            follower_right,
            follower_left,
            gripper_right,
            gripper_left,
            recorder,
            cameras,
        )

    finally:
        print("Shutting down...")
        try:
            for arm in (
                leader_right,
                leader_left,
                follower_right,
                follower_left,
            ):
                arm.torque_enable("group", "arm", False)
                arm.torque_enable("single", "gripper", False)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()


DATA COLLECTION SCRIPT

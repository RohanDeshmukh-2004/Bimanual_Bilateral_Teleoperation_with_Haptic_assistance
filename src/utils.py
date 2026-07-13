"""
robot_utils.py — ROS 2 interface helpers for ALOHA-style teleoperation.

Provides:
  ArmController   – low-level ROS 2 pub/sub wrapper for one interbotix arm.
  ImageRecorder   – thread-safe camera subscriber for multiple cameras.

All classes accept a shared rclpy.node.Node so they can coexist in one
process without requiring multiple spinning nodes.

FIXES vs original:
  - _call_service: no longer uses rclpy.spin_until_future_complete (deadlock).
    Uses a polling loop instead so the background MultiThreadedExecutor can
    resolve the future without conflict.
  - YUYV reshape: raw.reshape(H, W*2) instead of reshape(H, W, 2).
  - import time added at module level.
"""

import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg        import JointState, Image
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand
from interbotix_xs_msgs.srv import OperatingModes, TorqueEnable

from constants import GRIPPER_JOINT_NAME


# ─────────────────────────────────────────────────────────────────────────────
class ArmController:
    """
    Wraps one Interbotix arm (leader or follower) in a ROS 2 interface.

    Topics used
    -----------
    Sub:  /<robot_name>/joint_states                (sensor_msgs/JointState)
    Pub:  /<robot_name>/commands/joint_group        (JointGroupCommand)
    Pub:  /<robot_name>/commands/joint_single       (JointSingleCommand)
    Srv:  /<robot_name>/set_operating_modes         (OperatingModes)
    Srv:  /<robot_name>/torque_enable               (TorqueEnable)
    """

    def __init__(self, node: Node, robot_name: str, joint_names: list):
        self._node        = node
        self._robot_name  = robot_name
        self._joint_names = joint_names   # arm joints only (no gripper)
        self._n_joints    = len(joint_names)

        # Thread-safe joint state storage
        self._lock      = threading.Lock()
        self._js_pos    = None   # np.ndarray of all joint positions
        self._js_vel    = None
        self._js_effort = None
        self._js_names  = None   # joint name list as published

        # Publishers
        self._pub_group = node.create_publisher(
            JointGroupCommand,
            f"/{robot_name}/commands/joint_group",
            10,
        )
        self._pub_single = node.create_publisher(
            JointSingleCommand,
            f"/{robot_name}/commands/joint_single",
            10,
        )

        # Subscriber
        node.create_subscription(
            JointState,
            f"/{robot_name}/joint_states",
            self._js_callback,
            10,
        )

        # Service clients
        self._cli_modes = node.create_client(
            OperatingModes,
            f"/{robot_name}/set_operating_modes",
        )
        self._cli_torque = node.create_client(
            TorqueEnable,
            f"/{robot_name}/torque_enable",
        )

        node.get_logger().info(f"[ArmController] initialised: {robot_name}")

    # ── Callback ──────────────────────────────────────────────────────────────

    def _js_callback(self, msg: JointState):
        with self._lock:
            self._js_names  = list(msg.name)
            self._js_pos    = np.array(msg.position, dtype=np.float64)
            self._js_vel    = np.array(msg.velocity,  dtype=np.float64)
            self._js_effort = np.array(msg.effort,    dtype=np.float64)

    # ── Getters ───────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        with self._lock:
            return self._js_pos is not None

    def _read_state(self):
        """Return a consistent snapshot of (pos, vel, effort, names)."""
        with self._lock:
            return (self._js_pos.copy(),
                    self._js_vel.copy(),
                    self._js_effort.copy(),
                    list(self._js_names))

    def get_arm_qpos(self) -> np.ndarray:
        """Return arm joint positions in joint_names order (no gripper)."""
        pos, _, _, names = self._read_state()
        return np.array([pos[names.index(n)] for n in self._joint_names])


    def get_arm_qvel(self) -> np.ndarray:
        """Return arm joint velocities in joint_names order."""
        _, vel, _, names = self._read_state()
        return np.array([vel[names.index(n)] for n in self._joint_names])

    def get_arm_effort(self) -> np.ndarray:
        """Return arm joint efforts in joint_names order (for haptic feedback)."""
        _, _, effort, names = self._read_state()
        return np.array([effort[names.index(n)] for n in self._joint_names])

    def get_gripper_qpos(self) -> float:
        """Return the left_finger joint position (used as gripper state)."""
        pos, _, _, names = self._read_state()
        return float(pos[names.index(GRIPPER_JOINT_NAME)])

    def get_full_qpos(self) -> np.ndarray:
        """Return [arm joints..., gripper] — shape (n_joints+1,)."""
        return np.append(self.get_arm_qpos(), self.get_gripper_qpos())

    def get_full_qvel(self) -> np.ndarray:
        """Return [arm vels..., gripper_vel] — shape (n_joints+1,)."""
        _, vel, _, names = self._read_state()
        arm_vel  = np.array([vel[names.index(n)] for n in self._joint_names])
        grip_vel = float(vel[names.index(GRIPPER_JOINT_NAME)])
        return np.append(arm_vel, grip_vel)

    # ── Commands ──────────────────────────────────────────────────────────────

    def set_arm_positions(self, positions: np.ndarray):
        """Publish a joint group position command for the 'arm' group."""
        msg      = JointGroupCommand()
        msg.name = "arm"
        msg.cmd  = list(positions.astype(float))
        self._pub_group.publish(msg)

    def set_gripper_position(self, position: float):
        """
        Publish a single-joint position command for the gripper motor.
        position: left_finger angle in radians (position mode).
        """
        msg      = JointSingleCommand()
        msg.name = "gripper"
        msg.cmd  = float(position)
        self._pub_single.publish(msg)

    def set_arm_current(self, currents: np.ndarray):
        """
        Publish a joint group current command for the 'arm' group.
        currents: array of milliamps, one per arm joint.
        Used for haptic feedback on leader arms (current mode).
        """
        msg      = JointGroupCommand()
        msg.name = "arm"
        msg.cmd  = list(currents.astype(float))
        self._pub_group.publish(msg)

    # ── Service calls ─────────────────────────────────────────────────────────

    def _call_service(self, client, request, timeout: float = 5.0):
        """
        Call a ROS 2 service asynchronously and poll for completion.

        IMPORTANT: Do NOT use rclpy.spin_until_future_complete() here.
        The node is already being spun by a MultiThreadedExecutor in a
        background thread.  spin_until_future_complete() would conflict
        with that executor and cause a deadlock.

        Instead, call_async() and poll future.done() — the executor thread
        processes the service response and marks the future done.
        """
        if not client.wait_for_service(timeout_sec=timeout):
            self._node.get_logger().error(
                f"Service not available: {client.srv_name} ({self._robot_name})"
            )
            return None

        future = client.call_async(request)

        t0 = time.time()
        while not future.done():
            if time.time() - t0 > timeout:
                self._node.get_logger().error(
                    f"Service call timed out ({timeout}s): "
                    f"{client.srv_name} ({self._robot_name})"
                )
                return None
            time.sleep(0.005)   # 5 ms poll — fast enough, low CPU cost

        return future.result()

    def set_operating_mode(
        self,
        cmd_type: str,
        name: str,
        mode: str,
        profile_velocity: int = 131,
        profile_acceleration: int = 15,
    ):
        """
        Set the operating mode for a joint group or single joint.

        cmd_type : 'group' | 'single'
        name     : group name ('arm', 'gripper') or joint name
        mode     : 'position' | 'velocity' | 'pwm' | 'current' |
                   'linear_position' | 'current_based_position'
        """
        req = OperatingModes.Request()
        req.cmd_type             = cmd_type
        req.name                 = name
        req.mode                 = mode
        req.profile_type         = "velocity"    # True = velocity-based profile
        req.profile_velocity     = profile_velocity
        req.profile_acceleration = profile_acceleration
        self._call_service(self._cli_modes, req)
        self._node.get_logger().info(
            f"[{self._robot_name}] set {cmd_type}:{name} → {mode}"
        )

    def torque_enable(self, cmd_type: str, name: str, enable: bool):
        """Enable or disable torque for a group or joint."""
        req = TorqueEnable.Request()
        req.cmd_type = cmd_type
        req.name     = name
        req.enable   = enable
        self._call_service(self._cli_torque, req)
        state = "ON" if enable else "OFF"
        self._node.get_logger().info(
            f"[{self._robot_name}] torque {state} → {cmd_type}:{name}"
        )

    def go_to_home_pose(
        self,
        joint_positions: list,
        gripper_pos: float,
        wait: float = 3.0,
    ):
        """
        Command arm to a home pose.  Position mode must already be active.
        Blocks for *wait* seconds to allow the motion to complete.
        """
        self.set_arm_positions(np.array(joint_positions, dtype=float))
        self.set_gripper_position(gripper_pos)
        time.sleep(wait)


# ─────────────────────────────────────────────────────────────────────────────
class ImageRecorder:
    """
    Subscribes to multiple camera topics and stores the latest frame.

    Subscribes to:  /<cam_name>/image_raw   (sensor_msgs/Image)
    Frames are stored as BGR uint8 numpy arrays.
    """

    def __init__(self, node: Node, cam_names: list):
        self._cam_names = cam_names
        self._lock      = threading.Lock()
        self._frames    = {c: None for c in cam_names}

        for cam in cam_names:
            node.create_subscription(
                Image,
                f"/{cam}/image_raw",
                lambda msg, c=cam: self._img_callback(msg, c),
                2,   # small queue — only latest frame matters
            )
        node.get_logger().info(
            f"[ImageRecorder] subscribed to: {cam_names}"
        )

    def _img_callback(self, msg: Image, cam_name: str):
        """Convert ROS Image to numpy BGR uint8 and cache it."""
        enc = msg.encoding.lower()
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if enc in ("bgr8",):
            img = raw.reshape((msg.height, msg.width, 3))

        elif enc == "rgb8":
            img = raw.reshape((msg.height, msg.width, 3))[:, :, ::-1].copy()

        elif enc in ("yuv422", "yuyv"):
            # YUYV is 2 bytes per pixel packed.
            # OpenCV COLOR_YUV2BGR_YUYV expects shape (H, W*2) — NOT (H, W, 2).
            import cv2
            yuyv = raw.reshape(msg.height, msg.width * 2)
            img  = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)

        elif enc == "mono8":
            grey = raw.reshape((msg.height, msg.width))
            import cv2
            img  = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        else:
            # Unknown encoding: store raw and log a warning once
            self._node_logger_warn_once(
                f"ImageRecorder: unknown encoding '{enc}' for {cam_name}"
            )
            img = raw.reshape((msg.height, msg.width, -1))

        with self._lock:
            self._frames[cam_name] = img

    def _node_logger_warn_once(self, msg):
        # Avoids import of node just for logging; silently pass if unavailable
        pass

    def is_ready(self) -> bool:
        with self._lock:
            return all(f is not None for f in self._frames.values())

    def get_images(self) -> dict:
        """Return {cam_name: bgr_frame}.  Copies for thread safety."""
        with self._lock:
            return {c: f.copy() for c, f in self._frames.items()
                    if f is not None}

utility package robot_utils.py

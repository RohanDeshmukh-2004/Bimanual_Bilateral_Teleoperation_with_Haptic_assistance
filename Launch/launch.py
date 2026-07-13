# (aloha) cars@cars:~$ cat ~/interbotix_ws/src/aloha_rx200/launch/4arms_2cams.launch.py
"""
4arms_2cams.launch.py — Launch all 4 Interbotix arms and 2 eMeet webcams.

Hardware
--------
  Leaders   (RX200):  /dev/ttyDXL_leader_right   /dev/ttyDXL_leader_left
  Followers (VX300S): /dev/ttyDXL_follower_right  /dev/ttyDXL_follower_left
  Cameras:            /dev/CAM_HIGH               /dev/CAM_FRONT

Usage
-----
  source ~/interbotix_ws/install/setup.bash
  ros2 launch ~/interbotix_ws/src/aloha_rx200/launch/4arms_2cams.launch.py

Before launching:
  • Run setup_udev.sh once to create stable /dev/ttyDXL_* and /dev/CAM_* symlinks.
  • Copy the interbotix default motor config YAMLs into config/ and set the
    correct port in each one (see SETUP_GUIDE.md Step 6).
  • All four arms must be powered on and USB-connected.
  • Dynamixel Wizard must be fully closed.
  • No other application may be using the webcams.

Why motor_configs must be specified per arm
-------------------------------------------
The interbotix xs_sdk driver reads the serial port path from the
motor_configs YAML.  Without specifying it, all four arm nodes fall back
to the package default (e.g. /dev/ttyDXL) which does not exist on your
system.  Each arm needs its own YAML pointing to its udev symlink.
"""

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def arm_launch(robot_model: str, robot_name: str,
               motor_config: str, mode_config: str):
    """
    Helper: include xsarm_control.launch.py for one arm with explicit
    motor_configs (port) and mode_configs (operating modes).
    """
    interbotix_share = get_package_share_directory("interbotix_xsarm_control")
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(interbotix_share, "launch", "xsarm_control.launch.py")
        ),
        launch_arguments={
            "robot_model":   robot_model,
            "robot_name":    robot_name,
            "motor_configs": motor_config,   # ← port lives here
            "mode_configs":  mode_config,    # ← operating modes
            "use_rviz":      "false",
        }.items(),
    )


def camera_node(cam_name: str, video_device: str):
    """Helper: launch a usb_cam node for one camera."""
    return Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name=cam_name,
        namespace=cam_name,
        output="screen",
        parameters=[{
            "video_device":    video_device,
            "image_width":     640,
            "image_height":    480,
            "framerate":       30.0,
            # Try "mjpeg" if yuyv causes errors with your eMeet model.
            # Check supported formats: v4l2-ctl -d /dev/CAM_HIGH --list-formats-ext
            "pixel_format":    "yuyv",
            "camera_name":     cam_name,
            "camera_info_url": "",
            "io_method":       "mmap",
        }],
    )


def generate_launch_description():
    # ── Config directory (same package, one level up from launch/) ────────
    pkg_root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg          = os.path.join(pkg_root, "config")

    leader_mode_cfg   = os.path.join(cfg, "leader_mode.yaml")
    follower_mode_cfg = os.path.join(cfg, "follower_mode.yaml")

    # These are copies of the interbotix default motor configs with ONLY
    # the port field changed to match our udev symlinks.
    # See SETUP_GUIDE.md for how to create them.
    motor_leader_r  = os.path.join(cfg, "rx200_leader_right.yaml")
    motor_leader_l  = os.path.join(cfg, "rx200_leader_left.yaml")
    motor_follower_r = os.path.join(cfg, "vx300s_follower_right.yaml")
    motor_follower_l = os.path.join(cfg, "vx300s_follower_left.yaml")

    # ── Sanity check: fail early with a clear message if configs missing ──
    for path in [motor_leader_r, motor_leader_l,
                 motor_follower_r, motor_follower_l,
                 leader_mode_cfg, follower_mode_cfg]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n\nMissing config file: {path}\n"
                f"See SETUP_GUIDE.md → Step 6 to create per-arm motor configs.\n"
            )

    return LaunchDescription([
        # ── Leader arms (RX200) ───────────────────────────────────────────
        arm_launch("rx200", "rx200_leader_right",
                   motor_leader_r, leader_mode_cfg),

        arm_launch("rx200", "rx200_leader_left",
                   motor_leader_l, leader_mode_cfg),

        # ── Follower arms (VX300S) ────────────────────────────────────────
        arm_launch("vx300s", "vx300s_follower_right",
                   motor_follower_r, follower_mode_cfg),

        arm_launch("vx300s", "vx300s_follower_left",
                   motor_follower_l, follower_mode_cfg),

        # ── Cameras (staggered by 1 s to avoid USB enumeration races) ─────
        camera_node("cam_high",  "/dev/CAM_HIGH"),
        TimerAction(
            period=1.0,
            actions=[camera_node("cam_low", "/dev/CAM_FRONT")],
        ),
    ])

launch file

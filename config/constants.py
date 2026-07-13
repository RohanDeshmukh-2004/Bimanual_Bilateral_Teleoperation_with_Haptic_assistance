"""
constants.py — ALOHA-style static bimanual teleoperation
Hardware:  RX200 (leaders, 5-DOF)  +  VX300S (followers, 6-DOF)
Cameras:   2 × eMeet USB webcam (top-view = cam_high, front-view = cam_low)
"""
#constants.py — ALOHA-style static bimanual teleoperation
#Hardware:  RX200 (leaders, 5-DOF)  +  VX300S (followers, 6-DOF)
#Cameras:   2 × eMeet USB webcam (top-view = cam_high, front-view = cam_low)
#ROS:       ROS 2 Humble

#CRITICAL DOF MISMATCH NOTE
###--------------------------
#WX250S (original ALOHA leader) has 6 DOF: waist, shoulder, elbow,
 # forearm_roll, wrist_angle, wrist_rotate.
##RX200 (your leader) has 5 DOF:  waist, shoulder, elbow,
 # wrist_angle, wrist_rotate.           ← NO forearm_roll
#VX300S (your follower) has 6 DOF: same as WX250S.

#Strategy: forearm_roll on the VX300S is kept fixed at 0.0 during
#teleoperation.  The recorded dataset still has 7 values per arm
#(6 arm joints + 1 gripper), fully compatible with ACT state_dim=14.
#"""

# ── Timing ────────────────────────────────────────────────────────────────────
DT  = 0.02    # control loop period  (50 Hz)
FPS = 50      # camera recording rate

# ── Joint names ───────────────────────────────────────────────────────────────
LEADER_JOINT_NAMES = [
    "waist", "shoulder", "elbow",
    "wrist_angle", "wrist_rotate",          # RX200: 5 DOF
]

FOLLOWER_JOINT_NAMES = [
    "waist", "shoulder", "elbow",
    "forearm_roll",                          # VX300S only – held at 0
    "wrist_angle", "wrist_rotate",           # VX300S: 6 DOF
]

LEADER_NUM_JOINTS   = len(LEADER_JOINT_NAMES)    # 5
FOLLOWER_NUM_JOINTS = len(FOLLOWER_JOINT_NAMES)  # 6

# Gripper joint name as it appears in /joint_states
# Verify with:  ros2 topic echo /<robot_name>/joint_states --once
GRIPPER_JOINT_NAME = "gripper"

# ── DOF mapping: leader (RX200) ─► follower (VX300S) ─────────────────────────
# Each tuple: (leader_joint_index, follower_joint_index)
# forearm_roll (follower index 3) has no leader counterpart → stays 0.0
FOREARM_ROLL_FOLLOWER_IDX = 3       # index in FOLLOWER_JOINT_NAMES

LEADER_TO_FOLLOWER_MAP = [
    (0, 0),   # waist        → waist
    (1, 1),   # shoulder     → shoulder
    (2, 2),   # elbow        → elbow
              # (no leader)  → forearm_roll  [idx 3 – fixed at 0]
    (3, 4),   # wrist_angle  → wrist_angle
    (4, 5),   # wrist_rotate → wrist_rotate
]

# ── Robot names (must match robot_name in the launch file) ────────────────────
LEADER_RIGHT   = "rx200_leader_right"
LEADER_LEFT    = "rx200_leader_left"
FOLLOWER_RIGHT = "vx300s_follower_right"
FOLLOWER_LEFT  = "vx300s_follower_left"

# ── Gripper limits ────────────────────────────────────────────────────────────
# These values are for the standard Interbotix gripper.
# RX200 uses the same XL430-W250 gripper motor as WX250S.
# ⚠  Verify by slowly opening/closing the leader gripper and running:
#      ros2 topic echo /rx200_leader_right/joint_states --once
#    Record the left_finger values at fully open and fully closed positions.

# Leader (RX200) – measured at the left_finger joint
LEADER_RIGHT_GRIPPER_OPEN   =  1.3959
LEADER_RIGHT_GRIPPER_CLOSE  = -0.8406
FOLLOWER_RIGHT_GRIPPER_OPEN =  0.8559
FOLLOWER_RIGHT_GRIPPER_CLOSE= -0.7838
# Follower (VX300S) – measured at the left_finger joint
LEADER_LEFT_GRIPPER_OPEN    =  1.5000  # <--- YOU NEED TO UPDATE THIS ONE
LEADER_LEFT_GRIPPER_CLOSE   =  0.7225  
FOLLOWER_LEFT_GRIPPER_OPEN  =  0.8989  
FOLLOWER_LEFT_GRIPPER_CLOSE = -0.7639

# Position (metres, finger-tip separation) — used only for reference/viz
LEADER_GRIPPER_POS_OPEN    = 0.02417
LEADER_GRIPPER_POS_CLOSE   = 0.01244
FOLLOWER_GRIPPER_POS_OPEN  = 0.05800
FOLLOWER_GRIPPER_POS_CLOSE = 0.01844

# ── Gripper mapping lambdas ───────────────────────────────────────────────────
# Normalise leader gripper to [0, 1], then unnormalise to follower range.
#_L_RANGE = LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE
#_F_RANGE = FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE

#LEADER_GRIPPER_NORMALIZE_FN     = lambda x: (x - LEADER_GRIPPER_JOINT_CLOSE) / _L_RANGE
#FOLLOWER_GRIPPER_UNNORMALIZE_FN = lambda x: x * _F_RANGE + FOLLOWER_GRIPPER_JOINT_CLOSE

#LEADER2FOLLOWER_GRIPPER_FN      = lambda x: FOLLOWER_GRIPPER_UNNORMALIZE_FN(
 #                                     LEADER_GRIPPER_NORMALIZE_FN(x))

#FOLLOWER_GRIPPER_NORMALIZE_FN   = lambda x: (x - FOLLOWER_GRIPPER_JOINT_CLOSE) / _F_RANGE
#FOLLOWER_GRIPPER_MID             = (FOLLOWER_GRIPPER_JOINT_OPEN + FOLLOWER_GRIPPER_JOINT_CLOSE) / 2.0

# Threshold: when leader gripper (normalised) > this, teleoperation is enabled.
GRIPPER_ENABLE_THRESHOLD = 0.35   # ~35 % closed triggers start

# ── Start / home poses ────────────────────────────────────────────────────────
# These are the "ready" poses both arms move to before teleop.
# Adjust if they cause mechanical stress on your specific mounting.
LEADER_START_POSE   = [0.0, -0.96, 1.16, -0.3, 0.0]         # 5 joints (RX200)
FOLLOWER_START_POSE = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]    # 6 joints (VX300S)

LEADER_RIGHT_GRIPPER_HOME   = LEADER_RIGHT_GRIPPER_OPEN
LEADER_LEFT_GRIPPER_HOME    = LEADER_LEFT_GRIPPER_OPEN
FOLLOWER_RIGHT_GRIPPER_HOME = FOLLOWER_RIGHT_GRIPPER_OPEN
FOLLOWER_LEFT_GRIPPER_HOME  = FOLLOWER_LEFT_GRIPPER_OPEN
# ── Camera setup ──────────────────────────────────────────────────────────────
CAMERA_NAMES      = ["cam_high", "cam_low"]
CAMERA_IMG_HEIGHT = 480
CAMERA_IMG_WIDTH  = 640

# ── Task configs (consumed by record_episodes.py and ACT training) ────────────
TASK_CONFIGS = {
    "demo_task": {
        "dataset_dir":   "/home/user/aloha_data/demo_task",
        "episode_len":   300,         # timesteps at 50 Hz → 6 seconds
        "camera_names":  CAMERA_NAMES,
    },
    # Add more tasks here as needed.
}

Constants.py

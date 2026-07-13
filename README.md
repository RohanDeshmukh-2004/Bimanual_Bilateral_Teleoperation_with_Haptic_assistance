# Bimanual Bilateral Teleoperation with Failure-Aware Haptic Feedback

A low-cost bimanual bilateral teleoperation framework built on the ALOHA 
architecture using two ReactorX 200 (RX200) leader arms and two ViperX 300s 
(VX300s) follower arms on ROS 2 Humble with the Interbotix stack.

![Uploading WhatsApp Image 2026-07-07 at 11.48.32 AM.jpeg…]()


## What this project does

The operator moves the RX200 leader arms by hand. The VX300s follower arms 
mirror the motion in real time. Two additional feedback channels run 
simultaneously:

- **Haptic feedback** — follower actuator effort (contact forces) is filtered 
  and reflected back to the leader arms as physical resistance. The operator 
  feels what the robot is touching without any force-torque sensors.

- **Failure-aware corrective response** — a PID layer monitors 
  leader-follower tracking error per joint and injects blended corrective 
  commands when deviations exceed thresholds. After training, an ACT policy 
  runs in a background thread and applies policy-guided resistance to the 
  leader arms, steering the operator toward demonstration-consistent 
  trajectories.

## Hardware

| Role     | Robot        | DOF | Servo       |
|----------|-------------|-----|-------------|
| Leader   | RX200 × 2   | 5   | XL430-W250  |
| Follower | VX300s × 2  | 6   | XM430-W350  |
| Camera   | eMeet USB × 2 | — | 640×480 RGB |

## Key features

- 5-DOF to 6-DOF kinematic mapping with forearm roll held at zero
- Gripper scaling via linear normalisation across different motor ranges
- Asymmetric EMA haptic filter (α=0.92 onset, α=0.50 release)
- Per-joint deadband removal to eliminate gravity and noise artifacts
- PID corrective blend (α=0.35) with saturation guard and integral clamp
- HDF5 episode recording: proprioception, effort, haptic offsets, tracking 
  errors, failure flags, dual-camera RGB — ACT-compatible out of the box
- Background ACT inference thread (2.5 Hz CPU) with leader corrective PID
- 50 Hz control loop, 10 Hz gripper rate limiting for thermal protection

## Stack

- ROS 2 Humble
- Interbotix ROS 2 SDK
- Python 3.10
- PyTorch (ACT policy)
- HDF5 / h5py (episode storage)
- OpenCV (camera capture)

## Related work

This project builds on ALOHA, ACT, FACTR, and the failure-aware teleoperation 
framework of Zhou et al. (2026). The haptic design follows the guidance 
inaccuracy analysis of Smisek et al. (2015, IEEE WHC).

## Internship

Developed during a summer research internship at the Robotics Lab, 
IIT Dharwad (May–July 2026).

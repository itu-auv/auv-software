# auv-software
ITU AUV Team main software stack, on ROS 2 Jazzy and Gazebo Harmonic.

- [ROS2_PORT.md](ROS2_PORT.md): state of the ROS 2 port, how to build and run it, and
  how to port a ROS 1 branch.
- [BUGS.md](BUGS.md): bugs found during the port and left for separate fixes.

```bash
docker/dev/ros2 --build && docker/dev/ros2    # dev container (podman)
colcon build --symlink-install
ros2 launch auv_sim_bringup start_gazebo.launch.py gui:=true
```

## CI Status
| workflow     | status                                                                                                                                                                                                          |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| main ci      | [![Build Status](https://github.com/itu-auv/auv-software/actions/workflows/main_ci.yml/badge.svg)](https://github.com/itu-auv/auv-software/actions/workflows/main_ci.yml/badge.svg)                             |
| scheduled ci | [![Build Status](https://github.com/itu-auv/auv-software/actions/workflows/build_and_push_scheduled.yml/badge.svg)](https://github.com/itu-auv/auv-software/actions/workflows/build_and_push_scheduled.yml/badge.svg) |

# Bugs found during the ROS 2 port

Things noticed while porting that look wrong in the original code. The port keeps
them as they are unless the entry says **fixed**; fix them in their own PRs.

## Control

- `thruster_manager_ros.h`: the allocation result is used with `.value()` without
  checking that allocation succeeded.
- `thruster_manager_ros.h`: the per-thruster diagnostic wrench messages are built
  before the efforts are zeroed on wrench timeout, so they show stale thrust.
- `controller_ros.h`: `dt` passed to the PID was the loop rate in Hz, not seconds.
  **Fixed** in the port (Melih); `ki` and `integral_clamp_limits` in the YAML files
  were rescaled (x400 and /400 at 20 Hz) so behaviour is unchanged. Gains tuned on a
  ROS 1 branch need the same rescale when copied over.
- `AlignFrame.cfg` (`linear_kp`, `linear_kd`, `angular_kp`, `angular_kd`): nothing
  reads these gains.

## Localization

- DVL velocity and validity messages carry no header; `dvl_odometry_node` pairs them
  by arrival time.
- `pressure_odometry_node`: an out-of-range depth is clamped and published with the
  normal covariance, so the EKF trusts a value known to be wrong.

## Navigation

- `FollowPath` action: the goal's `paths` field is ignored. The server follows
  whatever arrives on `/planned_path` and never fills in feedback.
- `path_planner_node`: `/set_plan`, `/stop_planning` and `/planned_path` are absolute
  names, so they are not namespaced like the rest of the vehicle.
- `path_planners.path_for_bin` logs with the `[GatePathPlanner]` prefix.
- Unused: `follow_path_helpers.check_segment_progress`, `is_segment_completed`,
  `FollowPathActionClient.execute_paths`.

## Mapping

- `object_map_tf_server`: `dynamic_transform_callback` keeps one `static last_time`
  for all objects, so the `dt` given to an object's filter is the time since the
  last update of *any* object.
- `process_tracker_with_cloud` and `object_plane_fitter_node` each carry their own
  copy of the detection id → frame name table.

## Simulation

- Taluy Mini's base inertia matrix is not positive definite (Gazebo Harmonic rejects
  it; the sim model substitutes a sphere inertia).
- `robosub_bin` mesh has broken normals that crash collision checking; the Harmonic
  world uses a box as collision shape.
- `bar30` pressure sensor: the kPa-per-metre coefficient does not match
  `density * gravity` of the simulated water.

## Behaviour that changed because ROS 2 forced it

- A service handler can no longer make the call itself fail. Handlers that returned
  `False` in ROS 1 (e.g. `set_object_transform` when TF lookup fails) now answer
  with `success = false`; callers must check the field.
- `vision_msgs` detections carry the class as a string (`hypothesis.class_id`). The
  port keeps integer ids and writes them as decimal strings.
- There is no global parameter server. `waypoint_publisher` publishes the path
  summary it used to store at `/waypoint_gui/paths` as latched JSON on that topic.

## Vision

- `camera_detection_node`: the default `config_file` (`config/detection_objects.yaml`)
  does not exist; the node only starts when the launch file passes the real path.
- `yolo_result_bridge` and the tracker launch files use absolute topic names
  (`/yolo_result_*`, `/yolo_detections*`), so two vehicles cannot share a network.

## Hardware

- `auv_cam` (`auv_camera_ros.cpp`, `rtsp_camera_ros.cpp`): when the camera cannot be
  opened the code builds a `std::runtime_error` but never throws it, so the node
  carries on and exits on the first empty frame instead.
- `start_rtsp_camera.launch` passes `buffer_mode`, `drop_on_latency`,
  `reconnect_interval_sec` and `max_consecutive_failures`; `rtsp_cam_node` reads
  none of them.
- `realsense_camera.launch` forwarded about seventy arguments to the driver at their
  default values. The ROS 2 file keeps only the ones the vehicle changes.

## Missions (auv_smach)

- `bin.py`, second `__init__` using `bin_search_frame` (around line 436): the name
  is not a parameter of that constructor, so building that state machine raises
  `NameError`.
- `smach_monitor_node` used `rosnode kill joystick_node`. ROS 2 cannot kill a node
  by name, so the monitor now calls the `joystick_node/shutdown` service that the
  joystick node offers (see "Behaviour that changed").

## Simulation tools ported to Gazebo Harmonic

- `auv_sim_vision` read `/gazebo/model_states`. It now reads the world's
  `/world/<world>/pose/info` topic, bridged by its own launch files.
- `auv_sim/tf_gazebo_debug.py` used the Gazebo Classic spawn/delete/model-state
  services. It now uses the world's `create`/`remove` services and the robot's
  ground-truth odometry; `remove` is not bridged by the sim launch yet.

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
- ROS 2 smach decides the outcome of a `Concurrence` differently: of several matching
  `outcome_map` entries it takes the last, ROS 1 took the first. With the maps in
  `common.py` that turns "frame found while driving" from `succeeded` into
  `preempted`. `auv_smach.common.Concurrence` restores the ROS 1 rule.
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

Mission flow:

- `main_state_machine.py` builds `NavigateReturnThroughGateState` with a
  `gate_search_frame` argument the class does not take. `NAVIGATE_RETURN_THROUGH_GATE`
  is in the default `full_mission_states`, so the default full mission dies with a
  `TypeError` before the first state runs.
- `main_state_machine.py`: a state that is skipped while building (unknown name, or a
  `pathN` token with no path drawn in the GUI) is still the transition target of the
  state before it, so smach refuses to run the whole mission.
- The wrapper states that run an inner state machine (`InitializeState`,
  `NavigateThroughGateState`, `BinTaskState`, `OctagonTaskState`,
  `RandomPingerTaskState`, `NavigateReturnThroughGateState`, `RotateAroundBuoyState`)
  do not pass a preempt request on to it. Only `TorpedoTaskState` does. Nothing
  preempts the top-level machine today, so this only bites once something does.
- `DynamicPathWithTransformCheck` / `AlignFrameWithTransformCheck`: when the wait for
  the frame times out before the motion finishes, the result is `preempted`, and every
  caller maps `preempted` to "stop the mission". In `BinSearchSequenceState` both
  timers are 20 s and the frame wait starts first, so a search leg the vehicle cannot
  finish in 20 s ends the mission instead of moving to the next leg.
- `AimToProp`: `LOOK_AT` and `ALIGN_FRAME` get the same timeout and whichever ends
  first stops the other. If `LOOK_AT` wins the result is `preempted` (mission stops);
  if `ALIGN_FRAME` wins it is `succeeded`.
- `CheckAlignmentState` returns `succeeded` on timeout, so `AlignFrame` cannot report
  that the vehicle never got there (`CANCEL_ALIGNMENT_ON_FAIL` is unreachable).
  Torpedoes are launched and balls dropped after a timed-out alignment the same as
  after a good one.
- `SearchForPropState`: `CreateRotatingFrameState` returns `succeeded` after a full
  turn without ever seeing the frame, and `AimToProp` then times out into `succeeded`
  too. A search that finds nothing takes about a minute and looks like a success to
  the task (seen in the simulator: the gate task carried on without a gate).
- `SearchForPropState(rotation_speed=...)` only uses the speed to derive a period, and
  a non-positive speed silently becomes a 15 s turn. `octagon.py` passes `-0.2` to
  search the other way round and gets a turn in the usual direction at twice the
  speed.
- Several states return `preempted` without `service_preempt()`
  (`CreateRotatingFrameState`, `SetPlanState`, `PlanPathToSingleFrameState`,
  `PitchCorrection`, `RollTwoTimes`, `CheckBottleLinkState`), which leaves the request
  set on the state. `WaitForKillswitchEnabledState` and `DelayState` ignore preempts.
- `PickRemainingOctagonTargetsState._wait_for_fresh_targets` falls back to the last
  object list it ever received when no fresh one arrives in time.
- `RandomPingerTaskState` compares the pinger bearing against `torpedo_map_link_kde`
  and `octagon_link_kde` whatever `ignore_kde` says.
- `BinTaskState` ends by searching for and driving to the `odom` origin
  (`find_aim_return_home`, `RETURN_HOME_melih`), so the bin task also does the return
  home. It disables the bin trajectory publisher twice in a row.
- `SetRedBuoyRotationStartFrame` computes the facing angle in the buoy frame and
  publishes it in `odom` unrotated; right only while the buoy frame is aligned with
  `odom`. (`RotateAroundBuoyState` is not in the state map any more.)
- `AcousticTransmitter` publishes five times over five seconds and logs "published
  once".

Hard-coded vehicle names (break `taluy_mini` and any other namespace):

- `BallDropperSetAngleState` publishes on `/taluy/actuators/ball_dropper/set_angle`.
- `DynamicPathWithTransformCheck` defaults `align_source_frame` to `taluy/base_link`
  and no caller overrides it.
- `octagon.py`: `taluy/gripper_link_p2`, `taluy/base_link/dvl_mount_link`.
  `random_pinger.py`: `taluy/base_link`.

Dead or broken leftovers:

- `bin.py` `BinSecondTrialState`: uses `bin_search_frame`, which is not a parameter of
  its constructor (`NameError`), and transitions to a state that does not exist. Not
  used.
- Unused: `RotateAroundBuoyState` and the rest of `red_buoy.py`, `OctagonSurfaceState`
  unless `surface_and_return_if_octagon_first`, `CheckBottleLinkState`,
  `MoveGripperServiceState`, `OpenGripperServiceState`, `DropBallState`,
  `PlanGatePathsState`, `PublishGateAngleState`, `ResetOdometryPoseState`,
  `RotationState` outside `TwoYawState`, `NavigateToFrameState` outside `red_buoy.py`;
  the `torpedo_exit_angle`, `bin_front_look_depth`, `bin_bottom_look_depth`
  arguments; `legacy_target_selection`.
- State names left from debugging: `newhat`, `isim_degistir`, `lalala`, `ac_artik`,
  `yolo_bekle`, `SET_BATUHAN_DEPTH`, `RETURN_HOME_melih`.
- `smach_monitor_node` used `rosnode kill joystick_node`. ROS 2 cannot kill a node
  by name, so the monitor now calls the `joystick_node/shutdown` service that the
  joystick node offers (see "Behaviour that changed").

## Simulation tools ported to Gazebo Harmonic

- `auv_sim_vision` read `/gazebo/model_states`. It now reads the world's
  `/world/<world>/pose/info` topic. `ros_gz_bridge` drops the entity names when it
  converts that topic, so `auv_sim_bridge` has a small node for it
  (`world_pose_bridge`), started by the `auv_sim_vision` launch files.
- `auv_sim/tf_gazebo_debug.py` used the Gazebo Classic spawn/delete/model-state
  services. It now uses the world's `create`/`remove` services and the robot's
  ground-truth odometry; `remove` is not bridged by the sim launch yet.

## Gaps between the ROS 1 and the Harmonic simulator, found by running missions

- The Harmonic sim had no `sensors/dvl/enable` service (the ROS 1 mock had one), so
  `INITIALIZE` waited forever. **Fixed**: `sim_dvl_adapter.py` serves it and gates
  validity like the old mock.
- `sim_hardware_mocks.py` served `map/set_object_transform` as a stand-in while
  mapping was not ported; with the real `object_map_tf_server_node` running there
  were two servers for one service. **Fixed**: the stand-in is removed.
- The ROS 2 `robot_localization` node serves `enable` and `toggle` in its
  namespace, not under its own name as in ROS 1; missions and the GUI call
  `auv_localization_node/enable`. **Fixed**: both EKF launch files remap them.

## Small things

- `auv_smach/gate.py` defines `TransformServiceEnableState` and
  `PublishGateAngleState` twice (identical copies).
- `SetDepthState` documents a 30 s default timeout; it is 20 s.
- `reference_pose_publisher.py` imports `normalize_angle` from `angles` and then
  defines its own.
- Unused: `obj_type` in
  `competition_map_gui.py`, `before_clamp` in `multidof_pid_controller.h`, and in
  `process_tracker_with_cloud.cpp` the `distance` variable and five parameters of
  `createAndPublishDetection` (leftovers of removed 3D detection and marker output).
- `controller_ros.h`: members are initialised in a different order than declared
  (`-Wreorder`); harmless today, a trap if one initialiser starts using another.
- `auv_teleop/simulation_tab.py`: the "start detection" button launches
  `auv_detection tracker.launch`, which does not exist (the package has
  `tracker_all.launch` and `tracker_robosub.launch`).

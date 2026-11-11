# Bugs found during the ROS 2 port

Things that look wrong in the original code, found while porting it, reading it and
running missions in the simulator. The port keeps them as they are unless an entry
says **fixed**; fix them in their own PRs. Entries marked as port notes at the end
are not bugs: they record where ROS 2 forced a change.

## Worth fixing first

1. The default full mission could not start: `NavigateReturnThroughGateState` was
   built with an argument it did not take (Missions). **Fixed**.
2. A dynamic path to a frame that does not exist hangs the mission for ever
   (Navigation).
3. Timeouts inside `DynamicPathWithTransformCheck`, `AlignFrameWithTransformCheck` and
   `AimToProp` come out as `preempted`, which stops the whole mission (Missions).
4. A search that never sees its target, and an alignment that never converges, both
   report success (Missions).
5. The joystick node keeps sending the last stick position for ever after the
   controller disconnects (Teleoperation).
6. The PID integrals keep integrating while control is disabled (Control).
7. During a DVL dropout the velocity "measurement" is the EKF's own estimate
   (Localization).
8. The object map mixes up time steps between objects and extrapolates static objects
   with a velocity; the "KDE" frame is the mean of all detections (Mapping).

## Missions (auv_smach)

Mission flow:

- `main_state_machine.py` built `NavigateReturnThroughGateState` with a
  `gate_search_frame` argument the class did not take. `NAVIGATE_RETURN_THROUGH_GATE`
  is in the default `full_mission_states`, so the default full mission died with a
  `TypeError` before the first state ran (since #679, in ROS 1 too). **Fixed**: the
  class takes `gate_search_frame` and uses it in place of its hard-coded
  `gate_middle_part`, the same value the caller passes.
- `main_state_machine.py`: a state that is skipped while building (unknown name, or a
  `pathN` token with no path drawn in the GUI) is still the transition target of the
  state before it. smach's consistency check then fails and no state runs at all (the
  mission exits with outcome `None`).
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

- `roll.py` `TwoRollState`: DVL odometry is switched off before the roll and switched
  back on only on the success path. If `ROLL_TWO_TIMES` times out (15 s) the gate task
  aborts, the main state machine goes on to the next task (it does that for every
  `aborted`), and the rest of the mission runs without DVL odometry.

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
- `reference_pose_publisher.cmd_vel_callback` turns an angular velocity into a rotation
  by feeding `axis * angle` to `quaternion_from_euler`; that is only a rotation about
  that axis when one component is non-zero.
- `MultiDOFPIDController::control` runs every cycle, also while control is disabled or
  the kill switch is open, and nothing ever resets the integrals. They wind up (to the
  clamp limits, or without bound on an axis whose limit is 0) and are released when
  control comes back.
- `MultiDOFPIDController`: the integral clamp runs between the position-loop and the
  velocity-loop update, so the velocity integral is used one step before it is clamped.
- `controller_ros.h` `accel_callback` sets the angular accelerations to zero, so `kd` of
  the angular velocity loop multiplies nothing.
- `controller_ros.h` casts the loaded plugin to `SixDOFPIDController` and uses the
  result unchecked when publishing the desired velocity; any other controller plugin
  would crash the node.
- `MultiDOFPIDController` takes the attitude error as per-axis differences of Euler
  angles and uses them as body rates; only right near level attitude.

## Localization

- DVL velocity and validity messages carry no header; `dvl_odometry_node` pairs them
  by arrival time.
- `pressure_odometry_node`: an out-of-range depth is clamped and published with the
  normal covariance, so the EKF trusts a value known to be wrong.
- `dvl_odometry_node.compute_model_based_velocity`: once any odometry message has
  arrived (always), each step starts from the EKF's current velocity instead of the
  model's own state. The estimate is then fed to the same EKF as a measurement, so
  during a DVL dropout the filter is mostly measuring itself, and the "seed the model
  with the last valid DVL velocity" code has no effect.
- `dvl_odometry_node`: when the DVL mount transform is not available after 10 s it
  silently assumes a mount yaw of -45 degrees.
- `pressure_odometry_node.get_base_to_pressure_height` rotates a vector expressed in
  the sensor frame with the vehicle's orientation; right only while the sensor frame is
  not rotated against `base_link`.
- `imu_odometry_node`: the default `imu_calibration_path` is relative to the working
  directory, and a calibration file that exists but is malformed crashes the node.

## Navigation

- `FollowPath` action: the goal's `paths` field is ignored. The server follows
  whatever arrives on `/planned_path` and never fills in feedback.
- `path_planner_node`: `/set_plan`, `/stop_planning` and `/planned_path` are absolute
  names, so they are not namespaced like the rest of the vehicle.
- `path_planners.path_for_bin` logs with the `[GatePathPlanner]` prefix.
- Unused: `follow_path_helpers.check_segment_progress`, `is_segment_completed`,
  `FollowPathActionClient.execute_paths`.
- A dynamic path to a frame that does not exist hangs the mission: `/set_plan` answers
  success without checking the frame, the planner then publishes nothing, and the
  follow-path server waits for a path with no timeout (seen in the simulator: the gate
  task sat in `EXECUTE_PATH` until it was killed).
- `path_planners.straight_path_to_frame` catches every exception, so the TF retry
  branches in `path_for_gate` / `path_for_bin` never run. `path_for_bin` also defaults
  `bin_closer_frame` to `bin_close_trial`; the frame is called `bin_close_approach`.
  (Both planners are unused.)
- `generate_waypoints` ignores `interpolate_yaw` and `angular_diff` whenever
  `target_euler` is given, which is always.

## Mapping

- `object_map_tf_server`: `dynamic_transform_callback` keeps one `static last_time`
  for all objects, so the `dt` given to an object's filter is the time since the
  last update of *any* object.
- `process_tracker_with_cloud` and `object_plane_fitter_node` each carry their own
  copy of the detection id → frame name table.
- `kde_object_mapper` no longer estimates a density. The first "peak" of a class is the
  time-weighted mean of *all* its points, so one group of false detections elsewhere
  pulls the `*_kde` frame into open water between the groups. `min_peak_density` is
  compared with a weight sum that is 1 for the first peak. Parameter names and the
  docstring still describe a KDE.
- `kde_object_mapper`: a class with too few detections is published at its premap
  position under the same `*_kde` name, so missions cannot tell "seen" from "assumed".
- `octagon_trajectory_publisher` always builds its frames from `octagon_link_kde`, also
  when the mission runs with `ignore_kde`.
- `bin_trajectory_publisher`: `bin_exit` takes its depth and heading from
  `torpedo_target_realsense` (the variable and the log call it `gate_exit`), and waits
  1 s for it every cycle when the torpedo task has not run. The `second_trial_frame`
  parameter is read and then the name is hard-coded.
- `gate_trajectory_publisher`, `torpedo_frame_publisher`, `bin_trajectory_publisher`,
  `octagon_trajectory_publisher` look frames up at time "now" with a 4 s timeout, so one
  missing frame slows the whole loop to one pass per several seconds (the gate
  publisher's coin-flip frame shares that loop).
- `gate_trajectory_publisher`: `coin_flip_rescuer_pose` is computed and never used;
  `_shift_transform_parallel_to_gate_line` looks a transform up only to check its
  length.
- `ObjectPositionFilter` tracks objects that do not move with a constant-velocity
  Kalman filter and only predicts when a detection arrives. Detection noise becomes a
  velocity, and the next detection after a long gap first moves the frame by that
  velocity times the gap (made worse by the shared `last_time` above).
- `object_map_tf_server`: the unindexed frame name goes to whichever cluster of a class
  is nearest to the vehicle right now, so `gate_link` can change which object it means
  while the vehicle moves. Clusters are never dropped or limited in number; every false
  detection more than the threshold away from the others adds a frame until the map is
  cleared.
- `object_map_tf_server` holds the map lock across TF lookups with timeouts of up to
  4 s (`dynamic_transform_callback`), and broadcasting of every frame stops meanwhile.
- `process_tracker_with_cloud` `planeSegmentationAndPCA`: when the plane fit fails the
  PCA eigenvectors are used as a rotation matrix without the handedness check the
  plane path has; a left-handed set gives a meaningless quaternion.
- The premap (`config/premaps/premap.yaml`) holds positions in `odom`, so it is only
  right for one start pose, and `kde_object_mapper` throws away every detection more
  than `premap_max_distance` (5 m) from it. Started anywhere else, the `*_kde` frames
  stay at the premap positions however often the object is seen (in the simulator the
  octagon task drove 14 m to an empty spot). The launch files now take
  `premap:=none`.

## Vision

- `camera_detection_node`: the default `config_file` (`config/detection_objects.yaml`)
  does not exist; the node only starts when the launch file passes the real path.
- `yolo_result_bridge` and the tracker launch files use absolute topic names
  (`/yolo_result_*`, `/yolo_detections*`), so two vehicles cannot share a network.
- `segment_camera_handler._estimate_distance` calls the image-border checks without the
  camera's configured size; it works because the defaults (1920x1080) happen to match.
  `_build_rotation` looks up `taluy/base_link` whatever the vehicle is called, and
  `last_yaws` is keyed by class, so two objects of one class share a yaw memory.
- `front_camera_handler` / `kde_camera_handler._process_altitude_projection` require an
  altitude and then do not use it; the front one warns without throttling.
- `Prop.estimate_distance_diagonal` compares the pixel diagonal with the real *height*.
  Harmless for the slalom pipes (2.5 cm wide), wrong for anything wider.
- `camera_detection_node`: `set_front_camera_focus` does not affect the KDE handler,
  which reads the same detections.

## Teleoperation

- `joy_manager`: the last joystick message is used for ever. If the controller
  disconnects with a stick deflected, the node keeps publishing that velocity and
  `enable = true` at 50 Hz. There is no dead-man button and no staleness check.
- `joy_manager`: torpedo, dropper and reset-orientation service calls are made inside
  the joystick callback while holding the lock the publishing loop needs, so `cmd_vel`
  and `enable` pause for the length of the call (1 s or more when the service is
  missing, which is the controller's enable timeout).
- `joy_manager` `use_vel` mode zeroes the X/Y position gains of the controller and
  restores them on a clean shutdown only.
- `taluy_gui` simulation tab: the `buoy` test-state checkbox has no entry in the
  mission's `state_map`, so it is dropped without a message (ticked alone, the test
  mission is empty). `acoustic_tx`, `acoustic_rx`, `return` and `path1`-`path4` are
  in the map and have no checkbox. Same on ROS 1.
- `dynamic_roll_compensation` rotates the image about its centre pixel, not about the
  principal point of the camera, while the detections are later projected with the
  camera matrix.

## Hardware

- `auv_cam` (`auv_camera_ros.cpp`, `rtsp_camera_ros.cpp`): when the camera cannot be
  opened the code builds a `std::runtime_error` but never throws it, so the node
  carries on and exits on the first empty frame instead.
- `start_rtsp_camera.launch` passes `buffer_mode`, `drop_on_latency`,
  `reconnect_interval_sec` and `max_consecutive_failures`; `rtsp_cam_node` reads
  none of them.
- `realsense_camera.launch` forwarded about seventy arguments to the driver at their
  default values. The ROS 2 file keeps only the ones the vehicle changes.
- `wayfinder_dvl_node`: when the device time cannot be read the node logs the error
  and then crashes on `htime.ctime()`. On a bad beam it publishes `NaN` as altitude and
  range. `velocity` and `velocity_stamped` are advertised and never published.

## Simulation models

- Taluy Mini's base inertia matrix is not positive definite (Gazebo Harmonic rejects
  it; the sim model substitutes a sphere inertia).
- `robosub_bin` mesh has broken normals that crash collision checking; the Harmonic
  world uses a box as collision shape.
- `bar30` pressure sensor: the kPa-per-metre coefficient does not match
  `density * gravity` of the simulated water.

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

# Port notes

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
- The controller no longer writes its gains back to its YAML file (`save_parameters()`
  and the `config_file` parameter are gone), so values changed from `rqt` or by the
  gravity estimator last until the node restarts. Deliberate (Melih), to be revisited.
- There is no global parameter server. `waypoint_publisher` publishes the path
  summary it used to store at `/waypoint_gui/paths` as latched JSON on that topic.
- A ROS 2 service reply is delivered to every client of that service, and each client
  keeps only the last 10 it has not processed yet; a server does the same with
  requests. Nine nodes call `map/set_object_transform`, the frame publishers up to 80
  times a second, so a node that stalls for a fraction of a second loses its own
  reply, and a blocking call then waits for ever (seen once: the mission stopped in
  the octagon's last search). In ROS 1 a reply could not be lost. Python nodes that
  use `ros.init()` and the transform server now keep every request and reply, and
  `ServiceProxy` raises when the server disappears during a call. The cheaper design
  is the topic the server already has (`object_transform_updates`): a frame published
  20 times a second does not need an answer.

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
- The Harmonic launch always spawned the vehicle at the world origin, 14 m from the
  gate; the ROS 1 launch started the pool world in front of the gate. **Fixed**: `x`,
  `y`, `yaw` arguments, and `start:=gate|torpedo|bin|octagon|octagon_table|slalom` with
  the start poses the ROS 1 launch chose from its `objects` argument.
- Gazebo's 1 kHz `/clock` was bridged as is. **Fixed**: `clock_bridge` (see
  ROS2_PORT.md).
- The Harmonic vehicle had no depth camera, so the torpedo task stopped at
  `RESOLVE_TORPEDO_CLOSEST_FRAME`. **Fixed**: an `rgbd_camera` at the RealSense pose,
  bridged to the topics the ROS 1 plugin published (`camera/color/*`, `camera/depth/*`,
  `/points_filtered` via `depth_image_proc`).
- The Harmonic mock did not put the four octagon objects on the table at start-up (the
  ROS 1 mock did), spawned them all at one point when asked, and named them
  `mock_<item>_<n>`, which the simulated detectors do not recognise. **Fixed** in
  `sim_hardware_mocks.py`: spawned at start-up at random spots on the table, named as
  before (`pill`, `pill_2`, ...); `spawn_table_objects:=false` turns it off.
- The vehicle's buoyancy sphere (`base.urdf.xacro`) was meant not to touch anything,
  but URDF cannot carry `<surface>` and the conversion to SDF dropped the bitmask. The
  sphere reaches 39 cm above `base_link`, so at the gate depth of 0.7 m it passed
  1 cm under the plate hanging in the gate, or caught on it and stopped the mission
  there. **Fixed**: the launch sets the bitmask on the converted model.
- The simulator ran with the real pool's premap, which is in `odom` and so only means
  something from the place it was made for. Detections more than 5 m from it are
  thrown away, so tasks could not find their objects; with no premap, glimpses from
  far away put the bin 8 m from where it is and the full mission stopped there.
  **Fixed**: `premaps/sim_pool.yaml` for the pool world seen from the gate start; the
  sim launch uses it when starting there and no premap for the other start poses
  (`premap:=auto`).
- The roll through the gate (`roll:=true`) does not finish in the Harmonic simulator:
  the buoyancy sphere sits 0.2 m above the centre of mass, a righting moment of up to
  59 Nm against the 50 Nm the state asks for. The vehicle tips to about 60 degrees
  and `ROLL_TWO_TIMES` times out. The ROS 1 model had the same offset (54 Nm), so it
  may never have rolled there either. Not fixed; check the roll in water.

# ROS 2 port

`auv-software` on ROS 2 Jazzy with Gazebo Harmonic. The port is a translation of
the ROS 1 code, as close to line by line as ROS 2 allows: same packages, nodes,
topics, parameters and algorithms. Bugs found on the way are listed in
[BUGS.md](BUGS.md) and left alone unless keeping them was more work than fixing.

The earlier planning document and its agent hand-off log are in the git history
(`ROS2_JAZZY_MIGRATION_PLAN.md`).

## State

All 29 packages build with `colcon build`. The simulator launch starts the whole
stack (simulator, localization, control, mapping, navigation, vision, mission
support) and the mission launch runs a state machine against it.

Not done or not verified:

- Nothing has run on the vehicle. Camera, DVL, IMU and hydrophone drivers build but
  have not met their devices. `vectornav` is the upstream ROS 2 driver, not the
  team fork.
- In the simulator the gate, torpedo and bin tasks have been run from start to end;
  octagon, pinger and return have not.
- `docker/Dockerfile.auv*` and the image-building workflows still build ROS 1
  images (they wait on the choice of vehicle OS); `industrial_ci` builds on Jazzy.
- The simulator part of the port (Gazebo Classic + UUV simulator → Gazebo Harmonic)
  is a rewrite by necessity. The ROS 1 sim files it replaced are still in
  `auv_sim/` for reference and are not installed.

## Build and run

```bash
docker/dev/ros2 --build          # once: podman image with Jazzy, Gazebo, dependencies
docker/dev/ros2                  # shell in the container, workspace at /auv_ws
vcs import src < src/auv-software/third_party.repos
colcon build --symlink-install
ros2 launch auv_sim_bringup start_gazebo.launch.py gui:=true       # start:=torpedo|bin|... or x:= y:= yaw:=
ros2 launch auv_sim_vision tracker_robosub.launch                  # detections from the simulator
ros2 launch auv_smach start.launch use_sim_time:=true test_mode:=true "test_states:=['init']"
ros2 launch auv_bringup start.launch          # the real vehicle
```

## How the code was ported

Launch files are still XML (`*.launch`); `ros2 launch` reads them. The edits are
the ROS 2 spellings: `$(var x)`, `$(find-pkg-share pkg)`, `exec=`, and
`<push_ros_namespace>` inside `<group>`. The control, localization and simulator
launch files were ported to Python earlier and stay that way.

Parameter files loaded with `<param from=…>` are ROS 2 parameter files
(`/**:` or `/**/<node>:` then `ros__parameters:`). YAML files that code opens
itself (premaps, waypoints, detection objects) are unchanged.

There is no global parameter server, and so no global `/use_sim_time` either. The
simulator launch sets it for its own nodes; anything started separately against the
simulator needs `use_sim_time:=true` (the mission and teleop launch files take it).
A node of ours that finds a `/clock` while on wall time stops with an error instead
of running on mixed clocks.

There is no global parameter server. `start.launch` and the simulator launch push
`use_sim_time` and `environment.yaml` to every node they start; a node reads
`/env/pool_depth` as its own parameter `env.pool_depth`.

`.cfg` files became parameter lists next to the node that serves them
(`auv_common_lib.reconfigure.Server`); other nodes use `reconfigure.Client`.
`rqt_reconfigure` and `ros2 param set` edit them.

Python nodes use `auv_common_lib.ros`, which keeps the rospy habits the code was
written around:

| rospy | here |
|---|---|
| `rospy.init_node`, `rospy.spin` | `ros.init()`, `ros.spin(node[, loop])` |
| the implicit process-wide node (helper modules, SMACH states) | `ros.set_node(node)`, `ros.get_node()` |
| `rospy.get_param` (also nested dicts) | `get_param(node, name, default)` |
| `rospy.ServiceProxy` — callable, blocking, raises if nobody serves | `service_client(node, type, name)` |
| `rospy.Rate`, `rospy.on_shutdown`, `rospy.wait_for_message` | `Rate`, `on_shutdown`, `wait_for_message` |
| latched publisher | `latched_qos()` |

`ros.init()` also restores three rospy behaviours the code depends on. Every
subscriber, timer and service runs independently of the others (rclpy would run
all of a node's callbacks one at a time, so one slow callback stalls the rest).
And message fields are forgiving again. rclpy aborts the whole
process when a message with an `int` in a float field, a numpy scalar, or a
`Vector3` where a `Point` belongs is published; rospy accepted all of those and
the code relies on it in many places. And a TF listener receives transforms in a
thread of its own from the moment it is created (rclpy's waits for the node to
spin, so a lookup in a constructor would always time out).

## Porting a ROS 1 branch

`tools/ros1to2/` holds the scripts that did the mechanical part of this port.
They rewrite files in place and print what is left for a person:

```bash
tools/ros1to2/ros1to2.py [--node "get_node()"] file.py   # rospy calls -> rclpy / ros helpers
tools/ros1to2/nodeify.py file.py                         # class -> Node, main(), service callbacks
tools/ros1to2/ctors.py file.py                           # Point(1, 2, 3) -> Point(x=1, y=2, z=3)
tools/ros1to2/launch1to2.py file.launch                  # roslaunch XML -> ROS 2 launch XML
tools/ros1to2/ros1to2_cpp.py --node nh_ file.cpp         # includes, message types, logging, time
tools/ros1to2/wrapyaml.py config.yaml                    # rosparam YAML -> ROS 2 parameter file
tools/ros1to2/nodelint.py dir                            # run inside the container, see below
```

Things ROS 2 does differently that bit during the port:

- A `Node` subclass must not reuse the names `publishers`, `subscriptions`,
  `clients`, `services`, `timers`, `context`, `executor`, `handle`
  (`nodelint.py` finds them).
- Private names are `~/name`, not `~name`.
- `vision_msgs`: `bbox.center.position.x`, `results[0].hypothesis.class_id`
  (a string; we put the integer id in it).
- `CameraInfo` fields are lower case (`k`, `d`, `r`, `p`).
- Service handlers take `(request, response)` and cannot fail the call; report
  through the response.
- A subscriber only gets a latched message if it also asks for it
  (`latched_qos()` on both sides).
- Scripts started by launch files need the executable bit in git.
- PID integral gains tuned on ROS 1 need rescaling (see BUGS.md, Control).
- rclpy's `MultiThreadedExecutor` burns about 5 ms of CPU per message when the default
  QoS event handlers are attached to subscriptions (it spins while the callback waits
  for a worker thread). `ros.init()` turns those handlers off; the price is that
  incompatible-QoS warnings are no longer logged by our Python nodes.
- In simulation every rclpy node wakes up for each `/clock` message. Gazebo ticks at 1 kHz, so
  `auv_sim_bridge/clock_bridge` forwards the clock at 100 Hz of simulation time
  (`period` parameter). On the vehicle there is no `/clock` and an idle node costs
  about 1.5 % of a core.
- smach for ROS 2 resolves `Concurrence` outcomes differently from ROS 1 (see
  BUGS.md); use `auv_smach.common.Concurrence`.
- `ros_gz_bridge` drops entity names from `gz.msgs.Pose_V`; `world_pose_bridge` keeps
  them.

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
- Full competition missions have not been run in the new simulator.
- `rviz/araba.rviz` files were converted to RViz 2 plugin names but have not been
  opened yet.
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
ros2 launch auv_sim_bringup start_gazebo.launch.py gui:=true
ros2 launch auv_smach start.launch test_mode:=true "test_states:=['init']"
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

`ros.init()` also makes message fields forgiving again. rclpy aborts the whole
process when a message with an `int` in a float field, a numpy scalar, or a
`Vector3` where a `Point` belongs is published; rospy accepted all of those and
the code relies on it in many places.

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
- Scripts started by launch files need the executable bit in git.
- PID integral gains tuned on ROS 1 need rescaling (see BUGS.md, Control).

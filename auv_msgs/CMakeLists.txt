cmake_minimum_required(VERSION 3.16)
project(auv_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

## Declare action files
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/FollowPath.action"
  "msg/ObjectPose.msg"
  "msg/Keypoint.msg"
  "msg/KeypointResult.msg"
  "msg/PropsYaw.msg"
  "msg/SegmentMeasurement.msg"
  "msg/TDOA.msg"
  "msg/WaypointPath.msg"
  "srv/AlignFrameController.srv"
  "srv/CalibrateIMU.srv"
  "srv/PlanPath.srv"
  "srv/SendAcoustic.srv"
  "srv/SetDepth.srv"
  "srv/SetObjectTransform.srv"
  "srv/SetPremap.srv"
  "srv/SetDetectionFocus.srv"
  "srv/SpawnDebugCube.srv"
  "srv/SetDampingTarget.srv"
  "srv/SetWaypoint.srv"
  DEPENDENCIES geometry_msgs nav_msgs std_msgs
)

# # Generate added messages and services with any dependencies listed here
ament_export_dependencies(rosidl_default_runtime)

## Declare catkin package
ament_package()

cmake_minimum_required(VERSION 3.10.2)
project(auv_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  geometry_msgs
  message_generation
  nav_msgs
  std_msgs
)

## Declare action files
add_action_files(
  DIRECTORY action
  FILES
  FollowPath.action
)

add_message_files(
  FILES
  EulerOrientation.msg
  MotorCommand.msg
  ObjectPose.msg
  Power.msg
  PropsYaw.msg
  TDOA.msg
)

add_service_files(
  FILES
  AlignFrameController.srv
  PlanPath.srv
  SendAcoustic.srv
  SetDepth.srv
  SetObjectTransform.srv
  SetPremap.srv
  VisualServoing.srv
  SetDetectionFocus.srv
)

# # Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  nav_msgs
  std_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS
  actionlib
  actionlib_msgs
  geometry_msgs
  message_generation
  message_runtime
  nav_msgs
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

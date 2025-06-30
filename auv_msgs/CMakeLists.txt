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
  MotorCommand.msg
  Power.msg
  PropsYaw.msg
  VisualFeature.msg
)

add_service_files(
  FILES
  AlignFrameController.srv
  SetDepth.srv
  SetObjectTransform.srv
  VisualServoing.srv
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

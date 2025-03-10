cmake_minimum_required(VERSION 3.10.2)
project(auv_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  actionlib_msgs
  actionlib
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
)

add_service_files(
  FILES
  AlignFrameController.srv
  SetDepth.srv
  SetObjectTransform.srv
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
  nav_msgs
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS message_runtime actionlib_msgs actionlib geometry_msgs std_msgs nav_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

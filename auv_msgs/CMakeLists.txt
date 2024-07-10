cmake_minimum_required(VERSION 3.0.2)
project(auv_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  MotorCommand.msg
  Power.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs nav_msgs std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

cmake_minimum_required(VERSION 3.10.2)
project(auv_msgs)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  std_msgs
)

# # Generate messages in the 'msg' folder
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

# # Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs geometry_msgs message_generation message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

cmake_minimum_required(VERSION 3.10.2)
project(auv_canbus_msgs)

find_package(catkin REQUIRED COMPONENTS
  # geometry_msgs
  message_generation
  # nav_msgs
)


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Bool.msg
  DistanceMeasurement.msg
  Power.msg
  QuadDrivePulse.msg
  LatchedServo.msg
  SonarActivation.msg
  Pressure.msg
  Geometry.msg
)

# add_service_files(
#   FILES
#   AlignFrameController.srv
#   SetDepth.srv
#   SetObjectTransform.srv
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  # std_msgs
  # geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES auv_canbus_msgs
  CATKIN_DEPENDS
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

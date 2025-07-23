cmake_minimum_required(VERSION 3.10.2)
project(auv_localization)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  message_generation
  rospy
  std_msgs
)

add_service_files(
  FILES
  CalibrateIMU.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
  CATKIN_DEPENDS dynamic_reconfigure message_runtime std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/imu_odometry_node.py
  scripts/pressure_odometry_node.py
  scripts/dvl_odometry_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  config
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

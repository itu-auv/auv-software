cmake_minimum_required(VERSION 3.10.2)
project(auv_teleop)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  joy
  rospy
  std_msgs
  auv_msgs
  std_srvs
  message_generation
  message_runtime
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    sensor_msgs
    std_msgs
    auv_msgs
    std_srvs
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY scripts/
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
)

install(PROGRAMS
  scripts/joy_manager.py
  scripts/taluy_gui.py
  scripts/services_tab.py
  scripts/dry_test_tab.py
  scripts/vehicle_control_tab.py
  scripts/simulation_tab.py
  scripts/wrench_manager.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY images/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

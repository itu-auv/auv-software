cmake_minimum_required(VERSION 3.10.2)
project(auv_teleop)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  joy
  rospy
  Qt5
)

find_package(Qt5 COMPONENTS Widgets REQUIRED)

catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES acado_dynamics
  #  CATKIN_DEPENDS message_generation message_runtime roscpp
  #  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/joy_manager.py
  scripts/command_thread.py
  scripts/dry_test_tab.py
  scripts/services_tab.py
  scripts/taluy_gui.py
  scripts/vehicle_control_tab.py
  scripts/simulation_tab.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

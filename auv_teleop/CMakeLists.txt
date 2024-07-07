cmake_minimum_required(VERSION 3.0.2)
project(auv_teleop)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  joy
  rospy
)

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
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
  launch 
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}  
)

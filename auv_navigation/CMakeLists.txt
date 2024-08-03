cmake_minimum_required(VERSION 3.0.2)
project(auv_navigation)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  sensor_msgs
  rospy
  message_generation
  auv_common_lib
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES acado_dynamics
  #  CATKIN_DEPENDS message_generation message_runtime roscpp
  #  DEPENDS system_lib
  CATKIN_DEPENDS message_runtime std_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/goal_to_path.py
  scripts/path_follower.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY 
  launch 
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}  
)

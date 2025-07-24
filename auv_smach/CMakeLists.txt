cmake_minimum_required(VERSION 3.10.2)
project(auv_smach)

find_package(catkin REQUIRED COMPONENTS
  rospy
  smach
  smach_ros
  dynamic_reconfigure
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/SmachParameters.cfg
)

catkin_package(
  CATKIN_DEPENDS rospy smach smach_ros dynamic_reconfigure auv_navigation
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/main_state_machine.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

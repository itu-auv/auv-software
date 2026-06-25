cmake_minimum_required(VERSION 3.10.2)
project(auv_smach)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  auv_msgs
  geometry_msgs
  rospy
  smach
  smach_ros
  std_srvs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs auv_msgs geometry_msgs rospy smach smach_ros auv_navigation std_msgs std_srvs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/main_state_machine.py
  scripts/smach_monitor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

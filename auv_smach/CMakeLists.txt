cmake_minimum_required(VERSION 3.10.2)
project(auv_smach)

find_package(catkin REQUIRED COMPONENTS
  rospy
  smach
  smach_ros
  std_srvs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS rospy smach smach_ros auv_navigation std_srvs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/main_state_machine.py
  scripts/smach_monitor_node.py
  scripts/valve_test.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

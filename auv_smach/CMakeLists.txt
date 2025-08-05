cmake_minimum_required(VERSION 3.10.2)
project(auv_smach)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  rospy
  smach
  smach_ros
)

catkin_python_setup()

generate_dynamic_reconfigure_options(
  cfg/World.cfg
)

catkin_package(
  CATKIN_DEPENDS
    dynamic_reconfigure
    rospy
    smach
    smach_ros
    auv_navigation
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/main_state_machine.py
  scripts/environment_params_node.py
  scripts/smach_monitor_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

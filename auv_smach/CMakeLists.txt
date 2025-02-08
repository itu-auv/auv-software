cmake_minimum_required(VERSION 3.10.2)
project(auv_smach)

find_package(catkin REQUIRED COMPONENTS
  rospy
  smach
)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/main_state_machine.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cmake_minimum_required(VERSION 3.10.2)
project(auv_navigation)

find_package(catkin REQUIRED COMPONENTS)

catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
  scripts/follow_path_action_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

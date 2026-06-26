cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_bringup)

find_package(catkin REQUIRED)

catkin_package()

include_directories()

catkin_install_python(PROGRAMS
  scripts/episode_runner.py
  scripts/randomization.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

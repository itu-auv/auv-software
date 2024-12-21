cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_bringup)

find_package(catkin REQUIRED)

catkin_package()

include_directories()

install(DIRECTORY
  launch
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

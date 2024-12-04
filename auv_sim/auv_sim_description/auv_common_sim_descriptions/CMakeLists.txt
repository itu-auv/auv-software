cmake_minimum_required(VERSION 3.10.2)
project(auv_common_sim_descriptions)

find_package(catkin REQUIRED)

catkin_package()

include_directories()

install(DIRECTORY
  urdf
  worlds
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

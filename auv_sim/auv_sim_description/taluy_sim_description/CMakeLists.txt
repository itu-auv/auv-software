cmake_minimum_required(VERSION 3.10.2)
project(taluy_sim_description)

find_package(catkin REQUIRED)

catkin_package()

include_directories()

install(DIRECTORY
  urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

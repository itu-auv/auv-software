cmake_minimum_required(VERSION 3.10.2)
project(auv_common_descriptions)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY
  meshes
  urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

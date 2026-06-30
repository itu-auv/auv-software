cmake_minimum_required(VERSION 3.10.2)
project(taluy_mini_description)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY
  meshes
  rviz
  urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

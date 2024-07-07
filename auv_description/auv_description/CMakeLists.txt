cmake_minimum_required(VERSION 3.10.2)
project(auv_description)

find_package(catkin REQUIRED COMPONENTS)

catkin_package()

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
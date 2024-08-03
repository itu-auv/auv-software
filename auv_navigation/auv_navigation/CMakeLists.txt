cmake_minimum_required(VERSION 3.10.2)
project(auv_navigation)

find_package(catkin REQUIRED COMPONENTS
  auv_mapping
)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

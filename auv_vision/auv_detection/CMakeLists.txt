cmake_minimum_required(VERSION 3.0.2)
project(auv_detection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY 
  launch 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}  
)

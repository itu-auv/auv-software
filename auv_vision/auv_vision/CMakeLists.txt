cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

add_compile_options()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  auv_detection
  auv_common_lib
  rospy
  image_transport
  cv_bridge
  sensor_msgs
  std_msgs
  tf
)

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    auv_common_lib
    rospy
    sensor_msgs
    std_msgs
    tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dynamic_roll_compensation.py
    scripts/mapanything_socket.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

find_package(catkin REQUIRED COMPONENTS
  auv_detection
  rospy
  cv_bridge
  sensor_msgs
  tf
)

find_package(OpenCV 4 QUIET COMPONENTS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_videoio
  CONFIG)

if(NOT OpenCV_FOUND)
  message(STATUS "Did not find OpenCV 4, trying OpenCV 3")
  find_package(OpenCV 3 REQUIRED COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    CONFIG)
endif()

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    rospy
    sensor_msgs
    tf
  DEPENDS OpenCV
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dynamic_roll_compensation.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

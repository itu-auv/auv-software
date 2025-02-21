cmake_minimum_required(VERSION 3.10.2)
project(auv_hardware_bridge)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  image_transport
  rosserial_python
)

find_package(Boost REQUIRED COMPONENTS system)

set(_opencv_version 4)
find_package(OpenCV 4 QUIET)
if(NOT OpenCV_FOUND)
  message(STATUS "Did not find OpenCV 4, trying OpenCV 3")
  set(_opencv_version 3)
endif()


find_package(OpenCV ${_opencv_version} REQUIRED
  COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
  CONFIG
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES auv_cam
#  CATKIN_DEPENDS cv_bridge roscpp
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(serial_to_ros_bridge_node
  src/serial_to_ros_bridge_node.cpp
  src/serial_to_ros_bridge.cpp
)

add_executable(auv_gst_cam_node src/auv_gst_cam_node.cpp)

target_link_libraries(auv_gst_cam_node
  auv_camera_ros
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_library(auv_camera_ros src/auv_camera_ros.cpp)

target_link_libraries(auv_camera_ros
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_dependencies(serial_to_ros_bridge_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(serial_to_ros_bridge_node
  ${catkin_LIBRARIES}
  util
)

catkin_install_python(PROGRAMS
  scripts/wayfinder_dvl_node.py
  scripts/expansion_bridge_remapper_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS serial_to_ros_bridge_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS auv_camera_ros
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
        )


install(TARGETS auv_gst_cam_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )


install(DIRECTORY
        launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

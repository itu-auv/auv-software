cmake_minimum_required(VERSION 3.10.2)
project(auv_mapping)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  auv_msgs
  geometry_msgs
  roscpp
  tf2
  tf2_ros
)

find_package(OpenCV 4 QUIET COMPONENTS
  opencv_core
  opencv_imgproc
  opencv_imgcodecs
  opencv_video
  opencv_videoio
  CONFIG)

if(NOT OpenCV_FOUND)
  message(STATUS "Did not find OpenCV 4, trying OpenCV 3")
  find_package(OpenCV 3 REQUIRED COMPONENTS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_video
    opencv_videoio
    CONFIG)
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES object_position_filter
  CATKIN_DEPENDS roscpp tf2 tf2_ros geometry_msgs auv_msgs
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/prop_transform_publisher.py
  scripts/gate_trajectory_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(object_position_filter
  src/object_position_filter.cpp
)

target_link_libraries(object_position_filter
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(object_map_tf_server_node
  src/object_map_tf_server_ros.cpp
  src/object_map_tf_server_node.cpp
)

target_link_libraries(object_map_tf_server_node
  ${catkin_LIBRARIES}
  object_position_filter
)

install(TARGETS object_map_tf_server_node object_position_filter
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

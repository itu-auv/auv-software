cmake_minimum_required(VERSION 3.10.2)
project(auv_mapping)

# Debug sembolleri için build type ayarla
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

# Debug flag'lerini ekle
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  auv_msgs
  geometry_msgs
  roscpp
  tf2_ros
  octomap_ros
  octomap_msgs
  sensor_msgs
  message_filters
  pcl_ros
  pcl_conversions
  vision_msgs
  visualization_msgs
  image_geometry
  cv_bridge
  ultralytics_ros
)

find_package(PCL REQUIRED)

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
  CATKIN_DEPENDS roscpp tf2_ros geometry_msgs auv_msgs sensor_msgs message_filters pcl_ros pcl_conversions vision_msgs visualization_msgs image_geometry cv_bridge ultralytics_ros
  DEPENDS OpenCV PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
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

# Process Tracker With Cloud
add_executable(process_tracker_with_cloud
  src/process_tracker_with_cloud.cpp
)

target_link_libraries(process_tracker_with_cloud
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

install(TARGETS object_map_tf_server_node object_position_filter process_tracker_with_cloud
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

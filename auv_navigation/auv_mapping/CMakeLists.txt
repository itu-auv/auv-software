cmake_minimum_required(VERSION 3.10.2)
project(auv_mapping)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  auv_msgs
  cv_bridge
  geometry_msgs
  image_transport
  message_filters
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  shape_msgs
  std_msgs
  std_srvs
  tf2_geometry_msgs
  tf2_ros
  vision_msgs
  visualization_msgs
  dynamic_reconfigure
)

find_package(PCL REQUIRED COMPONENTS common segmentation)
find_package(Eigen3 REQUIRED)

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

generate_dynamic_reconfigure_options(
  cfg/SlalomTrajectory.cfg
  cfg/GateTrajectory.cfg
  cfg/TorpedoTrajectory.cfg
  cfg/GpsTargetFrame.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES object_position_filter
  CATKIN_DEPENDS roscpp rospy std_msgs std_srvs tf2_ros tf2_geometry_msgs geometry_msgs auv_msgs sensor_msgs message_filters pcl_ros pcl_conversions vision_msgs visualization_msgs shape_msgs cv_bridge image_transport dynamic_reconfigure
  DEPENDS OpenCV PCL
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
)

catkin_install_python(PROGRAMS
  scripts/gate_trajectory_publisher.py
  scripts/bin_trajectory_publisher.py
  scripts/torpedo_frame_publisher.py
  scripts/slalom_trajectory_publisher.py
  scripts/octagon_trajectory_publisher.py
  scripts/gps_target_frame_publisher.py
  scripts/acoustic_transmitter.py
  scripts/pipeline_trajectory_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_library(object_position_filter
  src/object_position_filter.cpp
)

target_link_libraries(object_position_filter
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

# Executables
add_executable(object_map_tf_server_node
  src/object_map_tf_server_ros.cpp
  src/object_map_tf_server_node.cpp
)

add_executable(process_tracker_with_cloud
  src/process_tracker_with_cloud.cpp
)

add_executable(realsense_filter_node
  src/realsense_filter.cpp
)

add_executable(object_plane_fitter_node
  src/object_plane_fitter_node.cpp
)

# Link libraries
target_link_libraries(object_map_tf_server_node
  ${catkin_LIBRARIES}
  object_position_filter
)

target_link_libraries(process_tracker_with_cloud
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(realsense_filter_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

target_link_libraries(object_plane_fitter_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS object_map_tf_server_node object_position_filter process_tracker_with_cloud realsense_filter_node object_plane_fitter_node
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

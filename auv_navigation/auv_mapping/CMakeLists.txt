cmake_minimum_required(VERSION 3.10.2)
project(auv_mapping)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  auv_msgs
  geometry_msgs
  roscpp
  rospy
  tf2
  tf2_ros
  tf2_geometry_msgs
  sensor_msgs
  std_srvs
  std_msgs
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
  CATKIN_DEPENDS
    roscpp
    tf2
    tf2_ros
    geometry_msgs
    auv_msgs
    sensor_msgs # Added based on find_package and common usage
    std_msgs    # Added based on find_package and common usage
    std_srvs    # Added based on find_package and common usage
    rospy       # Added because Python scripts are installed
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
  scripts/slalom_detector.py
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

add_executable(sonar_to_pointcloud_node
  src/sonar_to_pointcloud.cpp
)

# Link libraries for object_map_tf_server_node
target_link_libraries(object_map_tf_server_node
  ${catkin_LIBRARIES}
  object_position_filter
)

# Link libraries for sonar_to_pointcloud_node
target_link_libraries(sonar_to_pointcloud_node
  ${catkin_LIBRARIES}
)

install(TARGETS object_map_tf_server_node object_position_filter
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
# Also install the sonar_to_pointcloud_node executable
install(TARGETS sonar_to_pointcloud_node
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

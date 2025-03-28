cmake_minimum_required(VERSION 3.10.2)
project(auv_mapping)

add_compile_options(-std=c++17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(catkin REQUIRED COMPONENTS
  auv_msgs
  geometry_msgs
  roscpp
  tf2
  tf2_ros
  nav_msgs
)

# Find GTSAM
find_package(GTSAM REQUIRED)

catkin_package(
  CATKIN_DEPENDS auv_msgs geometry_msgs roscpp nav_msgs
  # DEPENDS gtsam # Removed based on catkin_lint feedback
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/prop_transform_publisher.py
  scripts/gate_trajectory_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(object_map_tf_server_node
  src/object_map_tf_server_node.cpp
)

# add_dependencies(object_map_tf_server_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(object_map_tf_server_node
  ${catkin_LIBRARIES}
)

add_executable(gtsam_slam_node
  src/gtsam_slam_node.cpp
)

target_link_libraries(gtsam_slam_node
  ${catkin_LIBRARIES}
  ${GTSAM_LIBRARIES}
)

install(TARGETS object_map_tf_server_node gtsam_slam_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

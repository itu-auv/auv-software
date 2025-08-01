cmake_minimum_required(VERSION 3.10.2)
project(auv_control)

add_compile_options(-std=c++17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  auv_common_lib
  auv_controllers
  auv_msgs
  dynamic_reconfigure
  geometry_msgs
  nav_msgs
  pluginlib
  roscpp
  rosfmt
  rospy
  tf2_ros
)

# Generate dynamic reconfigure options
generate_dynamic_reconfigure_options(
  cfg/Controller.cfg
  cfg/VisualServoing.cfg
  cfg/AlignFrame.cfg
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    auv_msgs
    dynamic_reconfigure
    geometry_msgs
    nav_msgs
    roscpp
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_install_python(PROGRAMS
  scripts/reference_pose_publisher.py
  scripts/battery_monitor_node.py
  scripts/align_frame_controller.py
  scripts/visual_servoing_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(${PROJECT_NAME}_node
  src/controller_node.cpp
  src/controller_ros.cpp
)

# Add dependencies to ensure dynamic reconfigure headers are generated before building
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(thruster_manager_node
  src/thruster_manager_node.cpp
  src/thruster_manager_ros.cpp
)

# Add dependencies to ensure dynamic reconfigure headers are generated before building
add_dependencies(thruster_manager_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(thruster_manager_node
  ${catkin_LIBRARIES}
)

install(TARGETS thruster_manager_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

cmake_minimum_required(VERSION 3.10.2)
project(auv_control)

add_compile_options(-std=c++17)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosfmt
  auv_msgs
  geometry_msgs
  auv_common_lib
  auv_controllers
  pluginlib
  nav_msgs
  tf2_ros
  dynamic_reconfigure
  rospy
)

# Generate dynamic reconfigure options
generate_dynamic_reconfigure_options(
  cfg/Controller.cfg
)

find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs
#  LIBRARIES auv_control
  # CATKIN_DEPENDS auv_controllers auv_common_lib
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/devel/include  # Ensure this is included
)

catkin_install_python(PROGRAMS
  scripts/depth_controller_node.py
  scripts/battery_monitor_node.py
  scripts/prop_transform_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(${PROJECT_NAME}_node
  src/controller_ros.cpp
  src/controller_node.cpp
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
  src/thruster_manager_ros.cpp
  src/thruster_manager_node.cpp
)

# Add dependencies to ensure dynamic reconfigure headers are generated before building
add_dependencies(thruster_manager_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

target_link_libraries(thruster_manager_node
  ${catkin_LIBRARIES}
)

install(TARGETS thruster_manager_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  launch
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

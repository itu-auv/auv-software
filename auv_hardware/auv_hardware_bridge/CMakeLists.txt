cmake_minimum_required(VERSION 3.0.2)
project(auv_hardware_bridge)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  rosserial_python
  roscpp
)

find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(serial_to_ros_bridge_node 
  src/serial_to_ros_bridge_node.cpp
  src/serial_to_ros_bridge.cpp
)

add_dependencies(serial_to_ros_bridge_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(serial_to_ros_bridge_node
  ${catkin_LIBRARIES}
  util
)

catkin_install_python(PROGRAMS
  scripts/wayfinder_dvl_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS serial_to_ros_bridge_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

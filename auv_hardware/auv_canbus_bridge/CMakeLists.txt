cmake_minimum_required(VERSION 3.0.2)
project(auv_canbus_bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  auv_msgs
  auv_canbus_msgs
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(auv_canbus_bridge_node src/auv_canbus_bridge_node.cpp)
target_link_libraries(auv_canbus_bridge_node ${catkin_LIBRARIES})

# catkin_install_python(PROGRAMS
#   scripts/send_can_msg.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

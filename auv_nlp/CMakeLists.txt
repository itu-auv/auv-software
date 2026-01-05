cmake_minimum_required(VERSION 3.0.2)
project(auv_nlp)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  auv_msgs
)

add_service_files(
  FILES
  Query.srv
  Execute.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  auv_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs auv_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
  scripts/nlp_commander_node.py
  scripts/gen_cli.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

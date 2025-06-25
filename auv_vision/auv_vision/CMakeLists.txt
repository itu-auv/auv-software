cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  message_filters
  auv_detection
)

catkin_package(CATKIN_DEPENDS
  rospy
  sensor_msgs
  cv_bridge
  message_filters
  auv_detection
  )

catkin_install_python(PROGRAMS scripts/merge_images_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

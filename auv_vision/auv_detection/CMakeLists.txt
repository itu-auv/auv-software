cmake_minimum_required(VERSION 3.10.2)
project(auv_detection)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  rospy
  sensor_msgs
  ultralytics_ros
  vision_msgs
)

catkin_package(
  CATKIN_DEPENDS cv_bridge geometry_msgs rospy sensor_msgs ultralytics_ros vision_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  scripts/rfdetr_onnx_node.py
  scripts/rfdetr_trt_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

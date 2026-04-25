cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

add_compile_options()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  auv_detection
  auv_msgs
  cv_bridge
  gazebo_msgs
  geometry_msgs
  image_transport
  message_filters
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_geometry_msgs
  tf2_ros
  vision_msgs
)

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    auv_msgs
    cv_bridge
    gazebo_msgs
    geometry_msgs
    image_transport
    message_filters
    nav_msgs
    rospy
    sensor_msgs
    std_msgs
    std_srvs
    tf
    tf2_geometry_msgs
    tf2_ros
    vision_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dynamic_roll_compensation.py
    scripts/yolo_result_bridge.py
    scripts/yolo_to_mask_node.py
    scripts/opencv_pipe_detector.py
    scripts/pipe_follower.py
    scripts/sim_bbox_node.py
    scripts/sim_keypoint_node.py
    scripts/depth_anything_trt_node.py
    scripts/depth_anything_zmq_client.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

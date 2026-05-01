cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

add_compile_options()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  auv_detection
  rospy
  image_transport
  cv_bridge
  sensor_msgs
  tf
  dynamic_reconfigure
  auv_common_lib
)

generate_dynamic_reconfigure_options(
  cfg/Aruco.cfg
)

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    rospy
    sensor_msgs
    tf
    dynamic_reconfigure
    auv_common_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/aruco_detection_node.py
    scripts/dynamic_roll_compensation.py
    scripts/yolo_result_bridge.py
    scripts/yolo_seg_to_mask.py
    scripts/pipe_follower_legacy.py
    scripts/sim_bbox_node.py
    scripts/sim_keypoint_node.py
    scripts/depth_anything_trt_node.py
    scripts/depth_anything_zmq_client.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

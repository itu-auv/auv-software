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
  geometry_msgs
  sensor_msgs
  std_msgs
  std_srvs
  tf
  dynamic_reconfigure
  ultralytics_ros
  vision_msgs
)
generate_dynamic_reconfigure_options(
  cfg/PipeFollowerLegacy.cfg
)

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    geometry_msgs
    rospy
    sensor_msgs
    std_msgs
    std_srvs
    tf
    dynamic_reconfigure
    ultralytics_ros
    vision_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dynamic_roll_compensation.py
    scripts/yolo_result_bridge.py
    scripts/yolo_seg_to_mask.py
    scripts/pipe_follower_legacy.py
    scripts/principal_point_overlay_node.py
    scripts/sim_bbox_node.py
    scripts/sim_tracker_node.py
    scripts/sim_keypoint_node.py
    scripts/depth_anything_trt_node.py
    scripts/depth_anything_sim_bridge_node.py
    scripts/depth_anything_zmq_client.py
    scripts/draw_mask_node.py
    scripts/reference_vector_generator_node.py
    scripts/dino_tracker_node.py
    scripts/vitpose_detection_node.py
    scripts/vitpose_seg_mask_node.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

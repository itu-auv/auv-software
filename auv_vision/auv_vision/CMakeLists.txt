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
)

catkin_package(
  CATKIN_DEPENDS
    auv_detection
    rospy
    sensor_msgs
    tf
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/dynamic_roll_compensation.py
    scripts/camera_detection_pose_estimator.py
    scripts/yolo_result_bridge.py
    scripts/yolo_image_annotator.py
    scripts/yolo_to_mask_node.py
    scripts/opencv_pipe_detector.py
    scripts/pipe_follower.py
    scripts/aruco_depth_estimation.py
  DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)

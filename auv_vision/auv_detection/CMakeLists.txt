cmake_minimum_required(VERSION 3.10.2)
project(auv_detection)

find_package(catkin REQUIRED COMPONENTS
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  scripts/yolo_result_bridge.py
  scripts/camera_detection_pose_estimator.py
  scripts/yolo_image_annotator.py
  scripts/yolo_to_mask_node.py
  scripts/pipe_trajectory_referance_pub.py
  scripts/pipe_follower_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

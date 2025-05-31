cmake_minimum_required(VERSION 3.10.2)
project(auv_detection)

find_package(catkin REQUIRED COMPONENTS
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  scripts/camera_detection_pose_estimator.py
  scripts/oak_yolo_ros_wrapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

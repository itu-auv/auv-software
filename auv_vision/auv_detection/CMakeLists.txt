cmake_minimum_required(VERSION 3.10.2)
project(auv_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS
  rospy
  cv_bridge
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  scripts/camera_detection_pose_estimator.py
  scripts/pipe_detector.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  scripts
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

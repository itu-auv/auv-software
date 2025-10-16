cmake_minimum_required(VERSION 3.10.2)
project(auv_rl_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
  auv_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    nav_msgs
    sensor_msgs
    visualization_msgs
    tf2_ros
    tf2_geometry_msgs
    auv_msgs
)

catkin_install_python(
  PROGRAMS
    scripts/train_agent.py
    scripts/test_components.py
    scripts/rl_agent_node.py
    scripts/plot_training_progress.py
    scripts/live_action_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  config
  worlds
  models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

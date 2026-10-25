cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_vision)

find_package(catkin REQUIRED)

catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/bbox_node.py
  scripts/tracker_node.py
  scripts/depth_anything_bridge_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

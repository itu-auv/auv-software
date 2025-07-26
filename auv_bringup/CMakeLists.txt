cmake_minimum_required(VERSION 3.10.2)
project(auv_bringup)

find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  rospy
)

generate_dynamic_reconfigure_options(
  cfg/SmachParameters.cfg
)

catkin_package(
  CATKIN_DEPENDS dynamic_reconfigure rospy
)

catkin_install_python(PROGRAMS
  scripts/smach_params_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install dynamic_reconfigure configuration files
install(DIRECTORY cfg/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg
)

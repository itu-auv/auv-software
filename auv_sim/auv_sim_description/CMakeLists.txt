cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_description)


find_package(catkin REQUIRED
#   rospy
#   roscpp
#   sensor_msgs
#   std_msgs
#   uuv_sensor_ros_plugins
#   uuv_sensor_ros_plugins_msgs
#   uuv_descriptions
#   uuv_gazebo_ros_plugins
#   uuv_assistants
#   uuv_descriptions
#   uuv_gazebo
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES auv_sim_description
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY 
  #launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}  
)

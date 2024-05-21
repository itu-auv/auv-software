cmake_minimum_required(VERSION 3.0.2)
project(auv_cmd_tools)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  rosserial_msgs
  auv_mainboard_bridge
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES auv_command_line_tools
#  CATKIN_DEPENDS rospy
#  DEPENDS system_lib
)


catkin_python_setup()

catkin_install_python(PROGRAMS scripts/auv 
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# requires autocomplete.sh
# catkin_add_env_hooks(autocomplete SHELLS sh DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)
catkin_add_env_hooks(autocomplete SHELLS zsh bash DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)

# catkin_add_env_hooks(env_variables SHELLS sh DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

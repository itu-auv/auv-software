cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_bridge)


find_package(catkin REQUIRED
  # roscpp
  # geometry_msgs
  # auv_msgs
  # uuv_gazebo_ros_plugins_msgs
  # uuv_sensor_ros_plugins_msgs
  # sensor_msgs
  # std_msgs
  # uuv_descriptions
  # uuv_gazebo_ros_plugins
  # uuv_assistants
  # uuv_descriptions
  # uuv_gazebo
  # ituauv_uuv_descriptions
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES auv_sim_bridge
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# add_executable(simulation_mock_node
#   src/simulation_mock_node.cpp
# )

# target_link_libraries(simulation_mock_node
#   ${catkin_LIBRARIES}
# )

# add_dependencies(simulation_mock_node 
#   ${simulation_mock_node_EXPORTED_TARGETS} 
#   ${catkin_EXPORTED_TARGETS}
# )

##
# Install
##

# catkin_install_python(PROGRAMS
#   scripts/blender_node.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(TARGETS simulation_mock_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

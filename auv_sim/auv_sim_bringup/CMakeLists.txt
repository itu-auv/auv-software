cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_bringup)


find_package(catkin REQUIRED

)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES auv_sim_bringup
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)

install(DIRECTORY 
  launch 
  config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}  
)
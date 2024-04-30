cmake_minimum_required(VERSION 3.0.2)
project(auv_logger)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rosbag
  roscpp
  rospy
)

catkin_package(
        CATKIN_DEPENDS
        roscpp
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(record
        src/record.cpp)

add_dependencies(record
        ${hydrophone_processing_EXPORTED_TARGETS}
        ${catkin_EXPORTED_TARGETS}
        )

target_link_libraries(record
        ${catkin_LIBRARIES}
        )

install(TARGETS record
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

install(DIRECTORY 
        launch 
        config
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}  
)
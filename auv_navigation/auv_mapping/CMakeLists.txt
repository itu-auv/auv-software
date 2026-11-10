cmake_minimum_required(VERSION 3.10.2)
project(auv_mapping)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-std=c++17)

find_package(ament_cmake REQUIRED)
find_package(auv_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs video videoio)

include_directories(include ${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(object_position_filter src/object_position_filter.cpp)
target_link_libraries(object_position_filter ${OpenCV_LIBRARIES})
ament_target_dependencies(object_position_filter geometry_msgs rclcpp tf2 tf2_geometry_msgs)

add_executable(object_map_tf_server_node
  src/object_map_tf_server_ros.cpp
  src/object_map_tf_server_node.cpp
)
target_link_libraries(object_map_tf_server_node object_position_filter)
ament_target_dependencies(object_map_tf_server_node auv_msgs geometry_msgs rclcpp std_srvs tf2 tf2_geometry_msgs tf2_ros)

add_executable(process_tracker_with_cloud src/process_tracker_with_cloud.cpp)
target_link_libraries(process_tracker_with_cloud ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})
ament_target_dependencies(process_tracker_with_cloud geometry_msgs message_filters pcl_conversions pcl_ros rclcpp sensor_msgs std_srvs tf2 tf2_ros vision_msgs visualization_msgs)

add_executable(realsense_filter_node src/realsense_filter.cpp)
target_link_libraries(realsense_filter_node ${PCL_LIBRARIES})
ament_target_dependencies(realsense_filter_node nav_msgs pcl_conversions rclcpp sensor_msgs std_msgs)

add_executable(object_plane_fitter_node src/object_plane_fitter_node.cpp)
target_link_libraries(object_plane_fitter_node ${PCL_LIBRARIES})
ament_target_dependencies(object_plane_fitter_node cv_bridge geometry_msgs message_filters pcl_conversions rclcpp sensor_msgs std_srvs tf2 tf2_ros vision_msgs)

install(PROGRAMS
  scripts/gate_trajectory_publisher.py
  scripts/bin_trajectory_publisher.py
  scripts/torpedo_frame_publisher.py
  scripts/octagon_trajectory_publisher.py
  scripts/acoustic_transmitter.py
  scripts/waypoint_gui.py
  scripts/waypoint_publisher.py
  scripts/premap_publisher.py
  scripts/competition_map_gui.py
  scripts/kde_object_mapper.py
  scripts/kde_visualizer.py
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS object_map_tf_server_node process_tracker_with_cloud realsense_filter_node object_plane_fitter_node
  DESTINATION lib/${PROJECT_NAME}
)
install(TARGETS object_position_filter
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()

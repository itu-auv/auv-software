cmake_minimum_required(VERSION 3.16)
project(auv_control)

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(auv_common_lib REQUIRED)
find_package(auv_controllers REQUIRED)
find_package(auv_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/reference_pose_publisher.py
  scripts/battery_monitor_node.py
  scripts/damping_tester.py
  scripts/gravity_compensation_estimator.py
  DESTINATION lib/${PROJECT_NAME}
)

add_executable(${PROJECT_NAME}_node src/controller_node.cpp src/controller_ros.cpp)
target_include_directories(${PROJECT_NAME}_node PRIVATE include)
target_link_libraries(${PROJECT_NAME}_node auv_common_lib::auv_common_lib auv_controllers::auv_controllers ${geometry_msgs_TARGETS} ${nav_msgs_TARGETS} pluginlib::pluginlib rclcpp::rclcpp ${std_msgs_TARGETS} tf2_ros::tf2_ros tf2_geometry_msgs::tf2_geometry_msgs)

add_executable(thruster_manager_node src/thruster_manager_node.cpp src/thruster_manager_ros.cpp)
target_include_directories(thruster_manager_node PRIVATE include)
target_link_libraries(thruster_manager_node Eigen3::Eigen fmt::fmt)
target_link_libraries(thruster_manager_node ${geometry_msgs_TARGETS} ${sensor_msgs_TARGETS} ${std_msgs_TARGETS} rclcpp::rclcpp tf2_ros::tf2_ros tf2_geometry_msgs::tf2_geometry_msgs)

install(TARGETS ${PROJECT_NAME}_node thruster_manager_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN ".svn" EXCLUDE)
ament_package()

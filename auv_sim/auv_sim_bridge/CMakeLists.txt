cmake_minimum_required(VERSION 3.16)
project(auv_sim_bridge)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(marine_acoustic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz-plugin REQUIRED COMPONENTS register)
find_package(gz_transport_vendor REQUIRED)
find_package(gz-transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)

add_library(auv-subsea-pressure-system SHARED src/subsea_pressure_system.cpp)
target_compile_features(auv-subsea-pressure-system PRIVATE cxx_std_17)
target_link_libraries(auv-subsea-pressure-system PRIVATE gz-sim::gz-sim gz-plugin::register)
install(TARGETS auv-subsea-pressure-system LIBRARY DESTINATION lib)

add_executable(world_pose_bridge src/world_pose_bridge.cpp)
target_compile_features(world_pose_bridge PRIVATE cxx_std_17)
target_link_libraries(world_pose_bridge gz-transport::gz-transport)
target_link_libraries(world_pose_bridge rclcpp::rclcpp ${tf2_msgs_TARGETS} ${geometry_msgs_TARGETS})

add_executable(clock_bridge src/clock_bridge.cpp)
target_compile_features(clock_bridge PRIVATE cxx_std_17)
target_link_libraries(clock_bridge gz-transport::gz-transport)
target_link_libraries(clock_bridge rclcpp::rclcpp ${rosgraph_msgs_TARGETS})
install(TARGETS world_pose_bridge clock_bridge DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/sim_hardware_mocks.py
  scripts/sim_pwm_to_force.py
  scripts/sim_dvl_adapter.py
  scripts/sim_pressure_depth_adapter.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY include/ DESTINATION include)
ament_package()

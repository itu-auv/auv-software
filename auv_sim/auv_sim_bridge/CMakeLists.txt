cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_bridge)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(marine_acoustic_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

add_library(auv-subsea-pressure-system SHARED src/subsea_pressure_system.cpp)
target_compile_features(auv-subsea-pressure-system PRIVATE cxx_std_17)
target_link_libraries(auv-subsea-pressure-system PRIVATE gz-sim8::gz-sim8 gz-plugin2::register)
install(TARGETS auv-subsea-pressure-system LIBRARY DESTINATION lib)

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

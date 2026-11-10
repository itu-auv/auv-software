cmake_minimum_required(VERSION 3.10.2)
project(auv_hydrophone)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/hydrophone_tdoa_to_angle_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

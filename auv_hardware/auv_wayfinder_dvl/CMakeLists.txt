cmake_minimum_required(VERSION 3.10.2)
project(auv_wayfinder_dvl)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/wayfinder_dvl_node.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

cmake_minimum_required(VERSION 3.16)
project(auv_sim)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/tf_gazebo_debug.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

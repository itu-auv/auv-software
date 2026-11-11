cmake_minimum_required(VERSION 3.16)
project(auv_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()

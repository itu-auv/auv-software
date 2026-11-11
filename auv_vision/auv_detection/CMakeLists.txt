cmake_minimum_required(VERSION 3.16)
project(auv_detection)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY models DESTINATION share/${PROJECT_NAME} OPTIONAL)

ament_package()

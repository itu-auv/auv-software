cmake_minimum_required(VERSION 3.16)
project(auv_bringup)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/damping_calculator.py
  scripts/image_saver.py
  scripts/smach_params_server.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()

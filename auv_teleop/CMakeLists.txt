cmake_minimum_required(VERSION 3.16)
project(auv_teleop)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/joy_manager.py
  scripts/taluy_gui.py
  scripts/services_tab.py
  scripts/dry_test_tab.py
  scripts/vehicle_control_tab.py
  scripts/simulation_tab.py
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch config images DESTINATION share/${PROJECT_NAME})

ament_package()

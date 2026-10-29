cmake_minimum_required(VERSION 3.10.2)
project(auv_sim_bringup)

find_package(ament_cmake REQUIRED)

include_directories()

install(PROGRAMS
  scripts/render_semi_final_pool_world.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch FILES_MATCHING PATTERN "*.py")
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

install(FILES ../../auv_bringup/config/ekf_taluy.yaml ../../auv_bringup/config/ekf_taluy_mini.yaml
  DESTINATION share/${PROJECT_NAME}/config)

ament_package()

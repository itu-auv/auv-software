cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/camera_detection_node.py
  scripts/dynamic_roll_compensation.py
  scripts/yolo_result_bridge.py
  scripts/depth_anything_trt_node.py
  scripts/depth_anything_zmq_client.py
  DESTINATION lib/${PROJECT_NAME}
)
# helper modules imported by camera_detection_node.py from its own directory
install(DIRECTORY scripts/handlers scripts/utils
  DESTINATION lib/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

ament_package()

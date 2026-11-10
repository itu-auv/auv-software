cmake_minimum_required(VERSION 3.10.2)
project(auv_cam)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

include_directories(include)

add_library(auv_camera_ros src/auv_camera_ros.cpp)
target_link_libraries(auv_camera_ros ${OpenCV_LIBRARIES})
ament_target_dependencies(auv_camera_ros cv_bridge image_transport rclcpp sensor_msgs)

add_library(rtsp_camera_ros src/rtsp_camera_ros.cpp)
target_link_libraries(rtsp_camera_ros ${OpenCV_LIBRARIES})
ament_target_dependencies(rtsp_camera_ros cv_bridge image_transport rclcpp sensor_msgs)

add_executable(auv_gst_cam_node src/auv_gst_cam_node.cpp)
target_link_libraries(auv_gst_cam_node auv_camera_ros)

add_executable(rtsp_cam_node src/rtsp_cam_node.cpp)
target_link_libraries(rtsp_cam_node rtsp_camera_ros)

install(TARGETS auv_camera_ros rtsp_camera_ros
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)
install(TARGETS auv_gst_cam_node rtsp_cam_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/oak_cam.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_package()

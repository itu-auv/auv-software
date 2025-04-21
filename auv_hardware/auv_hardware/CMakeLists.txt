cmake_minimum_required(VERSION 3.10.2)
project(auv_hardware)

find_package(catkin REQUIRED COMPONENTS
  auv_hardware_bridge
  auv_cam
)

catkin_package()

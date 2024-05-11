cmake_minimum_required(VERSION 3.0.2)
project(auv_hardware)

find_package(catkin REQUIRED COMPONENTS
  auv_hardware_bridge
)

catkin_package()

cmake_minimum_required(VERSION 3.10.2)
project(auv_vision)

find_package(catkin REQUIRED COMPONENTS
  auv_detection
)

catkin_package()

cmake_minimum_required(VERSION 3.8)
project(auv_msgs)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  # Messages
  "msg/EulerOrientation.msg"
  "msg/MotorCommand.msg"
  "msg/Power.msg"
  "msg/PropsYaw.msg"
  "msg/TDOA.msg"
  # Services
  "srv/AlignFrameController.srv"
  "srv/PlanPath.srv"
  "srv/SendAcoustic.srv"
  "srv/SetDepth.srv"
  "srv/SetDetectionFocus.srv"
  "srv/SetObjectTransform.srv"
  "srv/VisualServoing.srv"
  # Actions
  "action/FollowPath.action"
  # Dependencies
  DEPENDENCIES
    action_msgs
    geometry_msgs
    nav_msgs
    std_msgs
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

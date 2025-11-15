#include <optional>
#include <type_traits>
#include <typeinfo>

#include "auv_common_lib/ros/rosparam.h"

// TODO: ROS2 Migration - This file needs to be completely rewritten
// ROS2 uses a different parameter system without XmlRpc
// For now, providing stub implementations to allow compilation

namespace auv {
namespace common {
namespace rosparam {

// These are stub implementations that need to be properly implemented
// using ROS2's parameter system (rclcpp::Node::declare_parameter, get_parameter, etc.)

template <>
Eigen::Matrix<double, 6, 6> detail::parse<Eigen::Matrix<double, 6, 6>>(
    const rclcpp::Parameter &param) {
  // TODO: Implement using ROS2 parameter API
  throw std::runtime_error("Not yet implemented for ROS2");
}

template <>
Eigen::Matrix<double, 6, 1> detail::parse<Eigen::Matrix<double, 6, 1>>(
    const rclcpp::Parameter &param) {
  // TODO: Implement using ROS2 parameter API
  throw std::runtime_error("Not yet implemented for ROS2");
}

template <>
Eigen::Matrix<double, 12, 1> detail::parse<Eigen::Matrix<double, 12, 1>>(
    const rclcpp::Parameter &param) {
  // TODO: Implement using ROS2 parameter API
  throw std::runtime_error("Not yet implemented for ROS2");
}

}  // namespace rosparam
}  // namespace common
}  // namespace auv

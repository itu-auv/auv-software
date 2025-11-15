#pragma once

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "rclcpp/rclcpp.hpp"

// TODO: ROS2 Migration - This header needs significant changes
// XmlRpc is not used in ROS2. Parameters work differently.

namespace auv {
namespace common {
namespace rosparam {
namespace detail {

// TODO: Implement using ROS2 parameter API
template <typename T>
T parse(const rclcpp::Parameter& param);

}  // namespace detail

template <typename T>
struct parser {
  static T parse(const std::string param_name, rclcpp::Node::SharedPtr node) {
    // TODO: Implement using node->get_parameter()
    throw std::runtime_error("Not yet implemented for ROS2");
  }

  static T parse_or(const std::string param_name, rclcpp::Node::SharedPtr node,
                    const T& default_value) {
    // TODO: Implement using node->get_parameter_or()
    if (!node->has_parameter(param_name)) {
      return default_value;
    }
    return parse(param_name, node);
  }
};

}  // namespace rosparam
}  // namespace common
}  // namespace auv

#pragma once

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "rclcpp/rclcpp.hpp"

namespace auv {
namespace common {
namespace rosparam {
namespace detail {
inline rclcpp::Parameter get_parameter(const std::string& param_name,
                                       const rclcpp::Node& nh) {
  return nh.get_parameter(param_name);
}

template <typename T>
T parse(const rclcpp::Parameter& param);
}  // namespace detail

template <typename T>
struct parser {
  static T parse(const std::string param_name, const rclcpp::Node& nh) {
    const auto param = detail::get_parameter(param_name, nh);
    return detail::parse<T>(param);
  }

  static T parse_or(const std::string param_name, const rclcpp::Node& nh,
                    const T& default_value) {
    rclcpp::Parameter param;
    if (!nh.get_parameter(param_name, param) ||
        param.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
      return default_value;
    }

    return detail::parse<T>(param);
  }
};

}  // namespace rosparam
}  // namespace common
}  // namespace auv

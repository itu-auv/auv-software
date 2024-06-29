#pragma once

#include <Eigen/Core>
#include <type_traits>
#include <vector>

#include "ros/ros.h"

namespace auv {
namespace common {
namespace rosparam {
namespace detail {
inline XmlRpc::XmlRpcValue get_parameter(const std::string& param_name,
                                         const ros::NodeHandle& nh) {
  XmlRpc::XmlRpcValue param;

  const auto success = nh.getParam(param_name, param);

  ROS_ASSERT_MSG(success, "Failed to get parameter %s", param_name.c_str());

  return param;
}

template <typename T>
T parse(const XmlRpc::XmlRpcValue& param);
}  // namespace detail

template <typename T>
struct parser {
  static T parse(const std::string param_name, const ros::NodeHandle& nh) {
    const auto param = detail::get_parameter(param_name, nh);
    return detail::parse<T>(param);
  }

  static T parse_or(const std::string param_name, const ros::NodeHandle& nh,
                    const T& default_value) {
    const auto param = detail::get_parameter(param_name, nh);

    if (param.getType() == XmlRpc::XmlRpcValue::TypeInvalid) {
      return default_value;
    }

    return detail::parse<T>(param);
  }
};

}  // namespace rosparam
}  // namespace common
}  // namespace auv
#include <optional>
#include <type_traits>
#include <typeinfo>

#include "auv_common_lib/ros/rosparam.h"
#include "xmlrpcpp/XmlRpcException.h"
namespace {
template <typename T>
struct is_numerical_type
    : std::integral_constant<bool, std::is_floating_point<T>::value ||
                                       std::is_integral<T>::value> {};

template <typename T>
XmlRpc::XmlRpcValue::Type get_xmlrpc_type() {
  if (std::is_same<T, double>::value) {
    return XmlRpc::XmlRpcValue::TypeDouble;
  } else if (std::is_same<T, int>::value) {
    return XmlRpc::XmlRpcValue::TypeInt;
  } else if (std::is_same<T, bool>::value) {
    return XmlRpc::XmlRpcValue::TypeBoolean;
  } else if (std::is_same<T, std::string>::value) {
    return XmlRpc::XmlRpcValue::TypeString;
  } else {
    return XmlRpc::XmlRpcValue::TypeInvalid;
  }
}

template <typename T>
bool is_type(const XmlRpc::XmlRpcValue &param) {
  if constexpr (is_numerical_type<T>::value) {
    return param.getType() == XmlRpc::XmlRpcValue::TypeInt ||
           param.getType() == XmlRpc::XmlRpcValue::TypeDouble;
  }
  return param.getType() == get_xmlrpc_type<T>();
}

template <typename T>
std::optional<T> get_value(const XmlRpc::XmlRpcValue &param) {
  if (!is_type<T>(param)) {
    return std::nullopt;
  }

  try {
    return static_cast<T>(param);
  } catch (const XmlRpc::XmlRpcException &e) {
    try {
      return static_cast<T>(static_cast<int>(param));
    } catch (const XmlRpc::XmlRpcException &e) {
      return std::nullopt;
    }
  }
}

template <typename T, size_t N, size_t M>
inline Eigen::Matrix<T, N, M> get_parameter_as_matrix(
    const XmlRpc::XmlRpcValue &param) {
  using Matrix = Eigen::Matrix<T, N, M>;

  Matrix matrix;

  ROS_ASSERT_MSG(param.getType() == XmlRpc::XmlRpcValue::TypeArray,
                 "param: %s is not an array", param.toXml().c_str());
  ROS_ASSERT(param.size() == N);

  for (size_t i = 0; i < N; i++) {
    ROS_ASSERT(param[i].getType() == XmlRpc::XmlRpcValue::TypeArray);
    ROS_ASSERT(param[i].size() == M);

    for (size_t j = 0; j < M; j++) {
      const auto value = get_value<T>(param[i][j]);
      if (!value.has_value()) {
        ROS_FATAL_STREAM("param[" << i << "][" << j << "] is not of type "
                                  << typeid(T).name());
      }
      matrix(i, j) = *value;
    }
  }
  return matrix;
}

template <typename T, size_t N>
inline Eigen::Matrix<T, N, 1> get_parameter_as_vector(
    const XmlRpc::XmlRpcValue &param) {
  using Vector = Eigen::Matrix<T, N, 1>;

  Vector vector;

  ROS_ASSERT_MSG(param.getType() == XmlRpc::XmlRpcValue::TypeArray,
                 "param: %s is not an array", param.toXml().c_str());
  ROS_ASSERT(param.size() == N);

  for (size_t i = 0; i < N; i++) {
    ROS_ASSERT(param[i].getType() == XmlRpc::XmlRpcValue::TypeDouble ||
               param[i].getType() == XmlRpc::XmlRpcValue::TypeInt);

    vector(i) = static_cast<T>(param[i]);
  }
  return vector;
}

}  // namespace

namespace auv {
namespace common {
namespace rosparam {

template <>
Eigen::Matrix<double, 6, 6> detail::parse<Eigen::Matrix<double, 6, 6>>(
    const XmlRpc::XmlRpcValue &param) {
  return get_parameter_as_matrix<double, 6, 6>(param);
}

template <>
Eigen::Matrix<double, 6, 1> detail::parse<Eigen::Matrix<double, 6, 1>>(
    const XmlRpc::XmlRpcValue &param) {
  return get_parameter_as_vector<double, 6>(param);
}

template <>
Eigen::Matrix<double, 12, 1> detail::parse<Eigen::Matrix<double, 12, 1>>(
    const XmlRpc::XmlRpcValue &param) {
  return get_parameter_as_vector<double, 12>(param);
}

}  // namespace rosparam
}  // namespace common
}  // namespace auv
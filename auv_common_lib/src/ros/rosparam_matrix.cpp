#include <cmath>
#include <stdexcept>

#include "auv_common_lib/ros/rosparam.h"
namespace {
template <typename T, size_t N, size_t M>
inline Eigen::Matrix<T, N, M> get_parameter_as_matrix(
    const rclcpp::Parameter &param) {
  using Matrix = Eigen::Matrix<T, N, M>;

  Matrix matrix;
  const auto values = param.as_double_array();
  if (values.size() != N * M) {
    throw std::invalid_argument("Invalid matrix parameter length: " +
                                param.get_name());
  }
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < M; j++) {
      const auto value = values[i * M + j];
      if (!std::isfinite(value)) {
        throw std::invalid_argument("Non-finite matrix parameter: " +
                                    param.get_name());
      }
      matrix(i, j) = value;
    }
  }
  return matrix;
}

template <typename T, size_t N>
inline Eigen::Matrix<T, N, 1> get_parameter_as_vector(
    const rclcpp::Parameter &param) {
  return get_parameter_as_matrix<T, N, 1>(param);
}

}  // namespace

namespace auv {
namespace common {
namespace rosparam {

template <>
Eigen::Matrix<double, 6, 6> detail::parse<Eigen::Matrix<double, 6, 6>>(
    const rclcpp::Parameter &param) {
  return get_parameter_as_matrix<double, 6, 6>(param);
}

template <>
Eigen::Matrix<double, 6, 1> detail::parse<Eigen::Matrix<double, 6, 1>>(
    const rclcpp::Parameter &param) {
  return get_parameter_as_vector<double, 6>(param);
}

template <>
Eigen::Matrix<double, 12, 1> detail::parse<Eigen::Matrix<double, 12, 1>>(
    const rclcpp::Parameter &param) {
  return get_parameter_as_vector<double, 12>(param);
}

}  // namespace rosparam
}  // namespace common
}  // namespace auv

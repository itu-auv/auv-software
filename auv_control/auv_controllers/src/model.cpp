#include "auv_controllers/model.h"

#include "auv_common_lib/ros/rosparam.h"

namespace auv::common::rosparam {

auv::control::SixDOFModel parser<auv::control::SixDOFModel>::parse(
    const std::string param_name, const rclcpp::Node& nh) {
  constexpr auto kDOF = 6;
  auv::control::Model<kDOF> model;
  using MatrixT = Eigen::Matrix<double, kDOF, kDOF>;

  model.mass_inertia_matrix = parser<MatrixT>::parse(param_name + ".mass_inertia_matrix", nh);
  model.linear_damping_matrix = parser<MatrixT>::parse(param_name + ".linear_damping_matrix", nh);
  model.quadratic_damping_matrix =
      parser<MatrixT>::parse(param_name + ".quadratic_damping_matrix", nh);
  return model;
};

}  // namespace auv::common::rosparam::detail

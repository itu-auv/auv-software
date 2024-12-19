#include "auv_controllers/model.h"

#include "auv_common_lib/ros/rosparam.h"

namespace auv::common::rosparam::detail {

template <>
auv::control::Model<6> parse(const XmlRpc::XmlRpcValue& param) {
  constexpr auto kDOF = 6;
  auv::control::Model<kDOF> model;
  using MatrixT = Eigen::Matrix<double, kDOF, kDOF>;

  ROS_ASSERT_MSG(param.getType() == XmlRpc::XmlRpcValue::TypeStruct,
                 "param is not a struct");

  ROS_ASSERT_MSG(param.hasMember("mass_inertia_matrix"),
                 "mass_inertia_matrix not found in param");

  ROS_ASSERT_MSG(param.hasMember("linear_damping_matrix"),
                 "linear_damping_matrix not found in param");

  ROS_ASSERT_MSG(param.hasMember("quadratic_damping_matrix"),
                 "quadratic_damping_matrix not found in param");

  model.mass_inertia_matrix = parse<MatrixT>(param["mass_inertia_matrix"]);
  model.linear_damping_matrix = parse<MatrixT>(param["linear_damping_matrix"]);
  model.quadratic_damping_matrix =
      parse<MatrixT>(param["quadratic_damping_matrix"]);
  return model;
};

}  // namespace auv::common::rosparam::detail

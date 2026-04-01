#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "auv_controllers/controller_base.h"

namespace auv {
namespace control {

class SixDOFGeometricController : public SixDOFControllerBase {
 public:
  WrenchVector control(const SixDOFState& state, const SixDOFCommand& command,
                       const SixDOFStateDerivative& state_derivative,
                       double dt) override;

  void set_parameters(const SixDOFControllerParameters& parameters) override;

 private:
  static Eigen::Matrix3d rotation_matrix(const Eigen::Quaterniond& orientation);
  static Eigen::Vector3d vee(const Eigen::Matrix3d& skew_matrix);
  static Eigen::Quaterniond normalized(const Eigen::Quaterniond& orientation);

  Eigen::Vector3d compute_position_error_body(
      const SixDOFState& state, const SixDOFCommand& command) const;
  Eigen::Vector3d compute_rotation_error(const SixDOFState& state,
                                         const SixDOFCommand& command) const;
  SixDOFTwist compute_desired_twist(const SixDOFState& state,
                                    const SixDOFCommand& command) const;
  SixDOFTwist apply_velocity_saturation(const SixDOFTwist& desired_twist) const;
  WrenchVector compute_velocity_wrench(
      const SixDOFState& state, const SixDOFTwist& desired_twist,
      const SixDOFStateDerivative& state_derivative, double dt);
  Vector6d twist_to_vector(const SixDOFTwist& twist) const;
  SixDOFTwist vector_to_twist(const Vector6d& vector) const;
  void clamp_integral();

  Vector6d integral_{Vector6d::Zero()};
};

}  // namespace control
}  // namespace auv

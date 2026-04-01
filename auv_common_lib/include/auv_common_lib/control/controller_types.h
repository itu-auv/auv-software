#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace auv {
namespace control {

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

struct SixDOFPose {
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};

  void normalize_orientation() {
    if (orientation.norm() <= 1e-9) {
      orientation = Eigen::Quaterniond::Identity();
      return;
    }

    orientation.normalize();
  }
};

struct SixDOFTwist {
  Eigen::Vector3d linear{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular{Eigen::Vector3d::Zero()};
};

struct SixDOFState {
  SixDOFPose pose;
  SixDOFTwist twist;

  void normalize_orientation() { pose.normalize_orientation(); }
};

struct SixDOFCommand {
  SixDOFPose pose;
  SixDOFTwist twist_ff;

  void normalize_orientation() { pose.normalize_orientation(); }
};

struct SixDOFStateDerivative {
  Eigen::Vector3d linear_acceleration{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_acceleration{Eigen::Vector3d::Zero()};
};

struct SixDOFControllerParameters {
  Vector6d pose_kp{Vector6d::Zero()};
  Vector6d vel_kp{Vector6d::Zero()};
  Vector6d vel_ki{Vector6d::Zero()};
  Vector6d vel_kd{Vector6d::Zero()};
  Vector6d vel_integral_clamp_limits{Vector6d::Zero()};
  Vector6d max_velocity{Vector6d::Constant(1e6)};
  double gravity_compensation_z{0.0};
};

}  // namespace control
}  // namespace auv

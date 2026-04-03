#include "auv_controllers/sixdof_geometric_controller.h"

#include <pluginlib/class_list_macros.h>

#include <algorithm>

namespace auv {
namespace control {

Eigen::Quaterniond SixDOFGeometricController::normalized(
    const Eigen::Quaterniond& orientation) {
  if (orientation.norm() <= 1e-9) {
    return Eigen::Quaterniond::Identity();
  }

  return orientation.normalized();
}

Eigen::Matrix3d SixDOFGeometricController::rotation_matrix(
    const Eigen::Quaterniond& orientation) {
  return normalized(orientation).toRotationMatrix();
}

Eigen::Vector3d SixDOFGeometricController::vee(
    const Eigen::Matrix3d& skew_matrix) {
  return Eigen::Vector3d(skew_matrix(2, 1), skew_matrix(0, 2),
                         skew_matrix(1, 0));
}

void SixDOFGeometricController::set_parameters(
    const SixDOFControllerParameters& parameters) {
  SixDOFControllerBase::set_parameters(parameters);
  clamp_integral();
}

SixDOFGeometricController::WrenchVector SixDOFGeometricController::control(
    const SixDOFState& state, const SixDOFCommand& command,
    const SixDOFStateDerivative& state_derivative, double dt) {
  SixDOFState normalized_state = state;
  SixDOFCommand normalized_command = command;
  normalized_state.normalize_orientation();
  normalized_command.normalize_orientation();

  const auto desired_twist = apply_velocity_saturation(
      compute_desired_twist(normalized_state, normalized_command));
  return compute_velocity_wrench(normalized_state, desired_twist,
                                 state_derivative, dt);
}

Eigen::Vector3d SixDOFGeometricController::compute_position_error_body(
    const SixDOFState& state, const SixDOFCommand& command) const {
  const auto rotation = rotation_matrix(state.pose.orientation);
  return rotation.transpose() * (command.pose.position - state.pose.position);
}

Eigen::Vector3d SixDOFGeometricController::compute_rotation_error(
    const SixDOFState& state, const SixDOFCommand& command) const {
  const auto rotation = rotation_matrix(state.pose.orientation);
  const auto desired_rotation = rotation_matrix(command.pose.orientation);
  return 0.5 * vee(rotation.transpose() * desired_rotation -
                   desired_rotation.transpose() * rotation);
}

SixDOFTwist SixDOFGeometricController::compute_desired_twist(
    const SixDOFState& state, const SixDOFCommand& command) const {
  Vector6d desired_twist = twist_to_vector(command.twist_ff);

  const auto position_error = compute_position_error_body(state, command);
  const auto rotation_error = compute_rotation_error(state, command);

  desired_twist.head<3>() +=
      parameters().pose_kp.head<3>().cwiseProduct(position_error);
  desired_twist.tail<3>() +=
      parameters().pose_kp.tail<3>().cwiseProduct(rotation_error);

  return vector_to_twist(desired_twist);
}

SixDOFTwist SixDOFGeometricController::apply_velocity_saturation(
    const SixDOFTwist& desired_twist) const {
  Vector6d limited_twist = twist_to_vector(desired_twist);

  limited_twist = limited_twist.cwiseMin(parameters().max_velocity)
                      .cwiseMax(-parameters().max_velocity);

  return vector_to_twist(limited_twist);
}

SixDOFGeometricController::WrenchVector
SixDOFGeometricController::compute_velocity_wrench(
    const SixDOFState& state, const SixDOFTwist& desired_twist,
    const SixDOFStateDerivative& state_derivative, double dt) {
  const Vector6d current_twist = twist_to_vector(state.twist);
  const Vector6d desired_twist_vector = twist_to_vector(desired_twist);

  const Vector6d velocity_error = desired_twist_vector - current_twist;

  if (dt > 0.0) {
    integral_ += velocity_error * dt;
    clamp_integral();
  }

  Vector6d acceleration = Vector6d::Zero();
  acceleration.head<3>() = state_derivative.linear_acceleration;
  acceleration.tail<3>() = state_derivative.angular_acceleration;

  const Vector6d pid_output = parameters().vel_kp.cwiseProduct(velocity_error) +
                              parameters().vel_ki.cwiseProduct(integral_) -
                              parameters().vel_kd.cwiseProduct(acceleration);

  const Vector6d pid_force = model().mass_inertia_matrix * pid_output;

  const Vector6d damping_force =
      model().linear_damping_matrix * desired_twist_vector +
      model().quadratic_damping_matrix *
          desired_twist_vector.cwiseAbs().cwiseProduct(desired_twist_vector);

  WrenchVector wrench = pid_force + damping_force;

  Eigen::Vector3d gravity_force_global = Eigen::Vector3d::Zero();
  gravity_force_global.z() = parameters().gravity_compensation_z;
  wrench.head<3>() += rotation_matrix(state.pose.orientation).transpose() *
                      gravity_force_global;

  return wrench;
}

Vector6d SixDOFGeometricController::twist_to_vector(
    const SixDOFTwist& twist) const {
  Vector6d vector = Vector6d::Zero();
  vector.head<3>() = twist.linear;
  vector.tail<3>() = twist.angular;
  return vector;
}

SixDOFTwist SixDOFGeometricController::vector_to_twist(
    const Vector6d& vector) const {
  SixDOFTwist twist;
  twist.linear = vector.head<3>();
  twist.angular = vector.tail<3>();
  return twist;
}

void SixDOFGeometricController::clamp_integral() {
  for (Eigen::Index i = 0; i < integral_.size(); ++i) {
    const double limit = parameters().vel_integral_clamp_limits(i);
    if (limit <= 0.0) {
      continue;
    }

    integral_(i) = std::clamp(integral_(i), -limit, limit);
  }
}

}  // namespace control
}  // namespace auv

PLUGINLIB_EXPORT_CLASS(auv::control::SixDOFGeometricController,
                       auv::control::SixDOFControllerBase)

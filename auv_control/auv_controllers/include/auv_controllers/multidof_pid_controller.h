#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <vector>

#include "angles/angles.h"
#include "auv_controllers/controller_base.h"

namespace auv {
namespace control {

template <size_t N>
class MultiDOFPIDController : public ControllerBase<N> {
  using Base = ControllerBase<N>;
  using WrenchVector = typename Base::WrenchVector;
  using StateVector = typename Base::StateVector;
  using Vectornd = Eigen::Matrix<double, N, 1>;
  using Vector2nd = Eigen::Matrix<double, 2 * N, 1>;
  using Matrixnd = Eigen::Matrix<double, N, N>;
  using Matrix2nd = Eigen::Matrix<double, 2 * N, 2 * N>;

 public:
  void set_kp(const Vector2nd& kp) { kp_ = kp.asDiagonal(); }

  void set_ki(const Vector2nd& ki) { ki_ = ki.asDiagonal(); }

  void set_kd(const Vector2nd& kd) { kd_ = kd.asDiagonal(); }

  /**
   * @brief Set the max velocity limits for the position controller output
   *
   * @param limits Vector of max velocities (absolute values)
   */
  void set_max_velocity_limits(const Vectornd& limits) {
    max_velocity_limits_ = limits;
  }

  /**
   * @brief Set the integral clamp limits for anti-windup
   *
   * @param limits Vector of integral limits
   *               First N elements for position, last N for velocity
   */
  void set_integral_clamp_limits(const Vector2nd& limits) {
    integral_clamp_limits_ = limits;
  }

  /**
   * @brief Set the gravity compensation for z-axis
   *
   * @param compensation The gravity compensation value to be added to z-axis
   * wrench
   */
  void set_gravity_compensation_z(double compensation) {
    gravity_compensation_z_ = compensation;
  }

  /**
   * @brief Calculate the control output, in the form of a wrench
   *
   * @param state current velocity vector
   * @param desired_state desired velocity vector
   * @param d_state current acceleration vector
   * @param dt delta time
   * @return Vector control output(force and torque)
   */
  WrenchVector control(const StateVector& state,
                       const StateVector& desired_state,
                       const StateVector& d_state, const double dt) {
    const auto inverse_rotation_matrix = get_world_to_body_rotation(state);

    Vectornd pos_pid_output = Vectornd::Zero();
    {
      const auto position_state = state.head(N);
      const auto desired_position = desired_state.head(N);
      const auto velocity_state = d_state.head(N);

      Vectornd error = Vectornd::Zero();
      error.head(3) = desired_position.head(3) - position_state.head(3);
      error.head(3) = inverse_rotation_matrix * error.head(3);
      for (size_t i = 3; i < N; ++i) {
        error(i) = angles::shortest_angular_distance(position_state(i),
                                                     desired_position(i));
      }

      const auto p_term = kp_.template block<N, N>(0, 0) * error;

      integral_.head(N) += error * dt;

      // Apply integral clamping (anti-windup)
      if (integral_clamp_limits_.norm() > 0) {  // Check if limits are set
        for (size_t i = 0; i < 2 * N; ++i) {
          const double limit = integral_clamp_limits_(i);
          if (limit > 0) {  // Only clamp if limit is positive
            double before_clamp = integral_(i);
            integral_(i) = std::max(-limit, std::min(limit, integral_(i)));
          }
        }
      }

      const auto i_term = ki_.template block<N, N>(0, 0) * integral_.head(N);

      // d/dt (desired) is considered to be zero
      const auto d_term =
          kd_.template block<N, N>(0, 0) * (Vectornd::Zero() - velocity_state);

      pos_pid_output = p_term + i_term + d_term;
    }

    const auto velocity_state = state.tail(N);
    const auto desired_velocity = (desired_state.tail(N) + pos_pid_output)
                                      .cwiseMin(max_velocity_limits_)
                                      .cwiseMax(-max_velocity_limits_);
    const auto acceleration_state = d_state.tail(N);

    const auto error = desired_velocity - velocity_state;
    const auto p_term = kp_.template block<N, N>(N, N) * error;

    integral_.tail(N) += error * dt;
    const auto i_term = ki_.template block<N, N>(N, N) * integral_.tail(N);

    // d/dt (desired) is considered to be zero
    const auto d_term = kd_.template block<N, N>(N, N) *
                        (Vectornd::Zero() - acceleration_state);

    const auto pid_output = p_term + i_term + d_term;
    const auto mass_matrix = actual_mass_matrix(state);

    const auto pid_force = mass_matrix * pid_output;

    StateVector feedforward_state = desired_state;
    feedforward_state.tail(N) = desired_velocity;
    const auto damping_force = damping_control(feedforward_state);

    WrenchVector wrench = pid_force + damping_force;

    Eigen::Vector3d gravity_force_global = Eigen::Vector3d::Zero();
    gravity_force_global(2) = gravity_compensation_z_;
    wrench.head(3) += inverse_rotation_matrix * gravity_force_global;

    return wrench;
  }

 private:
  /**
   * @brief Compute the damping control term.
   *
   * @param state The current state of the system.
   * @return Vector The damping control term.
   *
   * damping control = D * state + Dq * |state| * state
   */
  Vectornd damping_control(const StateVector& state) const {
    const auto& velocity_state = state.tail(N);

    return this->model().linear_damping_matrix * velocity_state +
           this->model().quadratic_damping_matrix *
               velocity_state.cwiseAbs().cwiseProduct(velocity_state);
  }

  Matrixnd actual_mass_matrix(const StateVector& state) const {
    return this->model().mass_inertia_matrix;
  }

  Eigen::Matrix3d get_world_to_body_rotation(const StateVector& state) const {
    Eigen::AngleAxisd roll_angle(state[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(state[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(state[5], Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond rotation = yaw_angle * pitch_angle * roll_angle;
    // (body -> world).transpose() = world -> body
    return rotation.matrix().transpose();
  }

  // gains
  Vector2nd integral_{Vector2nd::Zero()};
  Vector2nd integral_clamp_limits_{
      Vector2nd::Zero()};  // Integral clamping limits
  Matrix2nd kp_;
  Matrix2nd ki_;
  Matrix2nd kd_;
  double gravity_compensation_z_{0.0};  // Gravity compensation for z-axis

  // Default to effectively unlimited (1e6)
  Vectornd max_velocity_limits_{Vectornd::Constant(1e6)};
};

using SixDOFPIDController = MultiDOFPIDController<6>;
}  // namespace control
}  // namespace auv

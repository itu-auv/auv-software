#pragma once
#include <Eigen/Core>
#include <array>
#include <vector>

#include "auv_controllers/controller_base.h"

namespace auv {
namespace control {

template <size_t N>
class MultiDOFPIDController : public ControllerBase<N> {
  using Base = ControllerBase<N>;
  using WrenchVector = typename Base::WrenchVector;
  using StateVector = typename Base::StateVector;
  using Vectornd = Eigen::Matrix<double, N, 1>;
  using Vector2nd = Eigen::Matrix<double, 2*N, 1>;
  using Matrixnd = Eigen::Matrix<double, N, N>;
  using Matrix2nd = Eigen::Matrix<double, 2*N, 2*N>;

 public:
  void set_kp(const Vector2nd& kp) { kp_ = kp.asDiagonal(); }

  void set_ki(const Vector2nd& ki) { ki_ = ki.asDiagonal(); }

  void set_kd(const Vector2nd& kd) { kd_ = kd.asDiagonal(); }

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
    Vectornd pos_pid_output = Vectornd::Zero();
    {
      const auto position_state = state.head(N);
      const auto desired_position = desired_state.head(N);
      const auto velocity_state = d_state.head(N);

      const auto error = desired_position - position_state;
      const auto p_term = kp_.template block<N, N>(0,0) * error;

      integral_.head(N) += error * dt;
      const auto i_term = ki_.template block<N, N>(0,0) * integral_.head(N);

      // d/dt (desired) is considered to be zero
      const auto d_term = kd_.template block<N, N>(0,0) * (Vectornd::Zero() - velocity_state);

      pos_pid_output = p_term + i_term + d_term;
    }

    const auto velocity_state = state.tail(N);
    const auto desired_velocity = desired_state.tail(N) + pos_pid_output;
    const auto acceleration_state = d_state.tail(N);

    const auto error = desired_velocity - velocity_state;
    const auto p_term = kp_.template block<N, N>(N,N) * error;

    integral_.tail(N) += error * dt;
    const auto i_term = ki_.template block<N, N>(N,N) * integral_.tail(N);

    // d/dt (desired) is considered to be zero
    const auto d_term = kd_.template block<N, N>(N,N) * (Vectornd::Zero() - acceleration_state);

    const auto pid_output = p_term + i_term + d_term;
    const auto mass_matrix = actual_mass_matrix(state);

    const auto pid_force = mass_matrix * pid_output;

    const auto damping_force = damping_control(desired_state);

    return pid_force + damping_force;
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

  // gains
  Vector2nd integral_{Vector2nd::Zero()};
  Matrix2nd kp_;
  Matrix2nd ki_;
  Matrix2nd kd_;
};

using SixDOFPIDController = MultiDOFPIDController<6>;
}  // namespace control
}  // namespace auv
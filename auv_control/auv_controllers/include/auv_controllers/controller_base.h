#pragma once
#include <Eigen/Core>
#include <array>
#include <vector>

#include "auv_controllers/model.h"

namespace auv {
namespace control {

template <size_t N>
class ControllerBase {
 public:
  using StateVector = Eigen::Matrix<double, 2U * N, 1>;
  using Matrix = Eigen::Matrix<double, N, N>;
  using Vector = Eigen::Matrix<double, N, 1>;
  using WrenchVector = Vector;

  /**
   * @brief Calculate the control output, in the form of a wrench
   *
   * @param state current state vector
   * @param desired_state desired state vector
   * @param state_derivative current derivative of state vector
   * @param dt delta time
   * @return Force and torque control output (wrench)
   */
  virtual WrenchVector control(const StateVector& state,
                               const StateVector& desired_state,
                               const StateVector& state_derivative,
                               const double dt) = 0;

  const Model<N>& model() const { return model_; }

  void set_model(const Model<N>& model) { model_ = model; }

 protected:
  Model<N> model_;
};

using SixDOFControllerBase = ControllerBase<6>;

}  // namespace control
}  // namespace auv
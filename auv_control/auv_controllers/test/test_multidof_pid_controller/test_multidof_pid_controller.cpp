#include <gtest/gtest.h>

#include "auv_controllers/multidof_pid_controller.h"
namespace {
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
constexpr auto kEpsilon = 1e-6;
}  // namespace

TEST(MultiDOFPIDController, TestSingleAxis) {
  using Vector = Eigen::Matrix<double, 1, 1>;
  using StateVector = Eigen::Matrix<double, 2, 1>;
  using Matrix = Eigen::Matrix<double, 1, 1>;
  auv::control::MultiDOFPIDController<1> controller;
  auv::control::Model<1> model;

  model.mass_inertia_matrix = Matrix::Identity() * 2;
  model.linear_damping_matrix = Matrix::Identity();
  model.quadratic_damping_matrix = Matrix::Identity() * 3;

  controller.set_model(model);

  controller.set_kp(Vector::Ones() * 1);
  controller.set_ki(Vector::Ones() * 2);
  controller.set_kd(Vector::Ones() * 3);

  const auto state = StateVector::Zero();
  const auto desired_state = StateVector::Ones();
  const auto d_state = StateVector::Ones();
  const auto dt = 0.1;

  const auto force_torque =
      controller.control(state, desired_state, d_state, dt);

  EXPECT_NEAR(force_torque(0), 0.4, kEpsilon);
}

TEST(MultiDOFPIDController, TestIdenticalAxis) {
  auv::control::MultiDOFPIDController<6> controller;
  auv::control::Model<6> model;

  model.mass_inertia_matrix = Matrix6d::Identity() * 2;
  model.linear_damping_matrix = Matrix6d::Identity();
  model.quadratic_damping_matrix = Matrix6d::Identity() * 3;

  controller.set_model(model);

  controller.set_kp(Vector6d::Ones() * 1);
  controller.set_ki(Vector6d::Ones() * 2);
  controller.set_kd(Vector6d::Ones() * 3);

  const auto state = Vector12d::Zero();
  const auto desired_state = Vector12d::Ones();
  const auto d_state = Vector12d::Ones();
  const auto dt = 0.1;

  const auto force_torque =
      controller.control(state, desired_state, d_state, dt);

  EXPECT_NEAR(force_torque(0), 0.4, kEpsilon);
  EXPECT_NEAR(force_torque(1), 0.4, kEpsilon);
  EXPECT_NEAR(force_torque(2), 0.4, kEpsilon);
  EXPECT_NEAR(force_torque(3), 0.4, kEpsilon);
  EXPECT_NEAR(force_torque(4), 0.4, kEpsilon);
  EXPECT_NEAR(force_torque(5), 0.4, kEpsilon);
}
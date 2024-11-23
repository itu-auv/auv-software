#include <gtest/gtest.h>

#include "auv_controllers/multidof_pid_controller.h"
namespace {
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
constexpr auto kEpsilon = 1e-6;
}  // namespace

TEST(MultiDOFPIDController, TestVelocityControlOnly) {
  auv::control::MultiDOFPIDController<6> controller;
  auv::control::Model<6> model;

  model.mass_inertia_matrix = Matrix6d::Identity() * 2;
  model.linear_damping_matrix = Matrix6d::Identity();
  model.quadratic_damping_matrix = Matrix6d::Identity() * 3;

  controller.set_model(model);

  // Vector6d ones and rest is zero
  Vector12d controller_gains = Vector12d::Zero();
  controller_gains.head(6) = Vector6d::Ones();

  controller.set_kp(controller_gains * 1);
  controller.set_ki(controller_gains * 2);
  controller.set_kd(controller_gains * 3);

  const auto state = Vector12d::Zero();
  const auto desired_state = Vector12d::Ones();
  const auto d_state = Vector12d::Ones();
  const auto dt = 0.1;

  const auto force_torque =
      controller.control(state, desired_state, d_state, dt);

  const auto expected_force = -2.72;

  EXPECT_NEAR(force_torque(0), expected_force, kEpsilon);
  EXPECT_NEAR(force_torque(1), expected_force, kEpsilon);
  EXPECT_NEAR(force_torque(2), expected_force, kEpsilon);
  EXPECT_NEAR(force_torque(3), expected_force, kEpsilon);
  EXPECT_NEAR(force_torque(4), expected_force, kEpsilon);
  EXPECT_NEAR(force_torque(5), expected_force, kEpsilon);
}
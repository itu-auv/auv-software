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

TEST(MultiDOFPIDController, LimitsAccelerationCommand) {
  auv::control::MultiDOFPIDController<6> controller;
  auv::control::Model<6> model;

  model.mass_inertia_matrix = Matrix6d::Identity();
  model.linear_damping_matrix = Matrix6d::Zero();
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);

  Vector12d controller_gains = Vector12d::Zero();
  controller_gains.tail(6) = Vector6d::Ones();
  controller.set_kp(controller_gains);
  controller.set_ki(Vector12d::Zero());
  controller.set_kd(Vector12d::Zero());
  controller.set_max_acceleration_limits(Vector6d::Constant(2.0));

  const auto state = Vector12d::Zero();
  Vector12d desired_state = Vector12d::Zero();
  desired_state.tail(6) = Vector6d::Constant(10.0);
  const auto d_state = Vector12d::Zero();

  const auto force_torque =
      controller.control(state, desired_state, d_state, 0.1);

  EXPECT_NEAR(force_torque(0), 2.0, kEpsilon);
  EXPECT_NEAR(force_torque(1), 2.0, kEpsilon);
  EXPECT_NEAR(force_torque(2), 2.0, kEpsilon);
  EXPECT_NEAR(force_torque(3), 2.0, kEpsilon);
  EXPECT_NEAR(force_torque(4), 2.0, kEpsilon);
  EXPECT_NEAR(force_torque(5), 2.0, kEpsilon);
}

TEST(MultiDOFPIDController, LimitsAccelerationCommandRate) {
  auv::control::MultiDOFPIDController<6> controller;
  auv::control::Model<6> model;

  model.mass_inertia_matrix = Matrix6d::Identity();
  model.linear_damping_matrix = Matrix6d::Zero();
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);

  Vector12d controller_gains = Vector12d::Zero();
  controller_gains.tail(6) = Vector6d::Ones();
  controller.set_kp(controller_gains);
  controller.set_ki(Vector12d::Zero());
  controller.set_kd(Vector12d::Zero());
  controller.set_max_acceleration_rate_limits(Vector6d::Constant(1.0));

  const auto state = Vector12d::Zero();
  Vector12d desired_state = Vector12d::Zero();
  desired_state.tail(6) = Vector6d::Constant(10.0);
  const auto d_state = Vector12d::Zero();

  const auto first_force_torque =
      controller.control(state, desired_state, d_state, 0.5);
  const auto second_force_torque =
      controller.control(state, desired_state, d_state, 0.5);

  EXPECT_NEAR(first_force_torque(0), 0.5, kEpsilon);
  EXPECT_NEAR(first_force_torque(5), 0.5, kEpsilon);
  EXPECT_NEAR(second_force_torque(0), 1.0, kEpsilon);
  EXPECT_NEAR(second_force_torque(5), 1.0, kEpsilon);
}

TEST(MultiDOFPIDController, AppliesYawCrossCouplingFeedforward) {
  auv::control::MultiDOFPIDController<6> controller;
  auv::control::Model<6> model;

  model.mass_inertia_matrix = Matrix6d::Zero();
  model.linear_damping_matrix = Matrix6d::Zero();
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);
  controller.set_kp(Vector12d::Zero());
  controller.set_ki(Vector12d::Zero());
  controller.set_kd(Vector12d::Zero());
  controller.set_yaw_cross_coupling_gain(4.0);

  const auto state = Vector12d::Zero();
  Vector12d desired_state = Vector12d::Zero();
  desired_state(6) = 2.0;
  desired_state(7) = 3.0;
  const auto d_state = Vector12d::Zero();

  const auto force_torque =
      controller.control(state, desired_state, d_state, 0.1);

  EXPECT_NEAR(force_torque(5), 24.0, kEpsilon);
}
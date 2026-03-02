#include <gtest/gtest.h>

#include "auv_controllers/multidof_pid_controller.h"
namespace {
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector3d = Eigen::Vector3d;
constexpr auto kEpsilon = 1e-6;
}  // namespace

TEST(MultiDOFPIDController, TestVelocityControlOnly) {
  auv::control::MultiDOFPIDController<6> controller;
  auv::control::Model<6> model;

  model.mass_inertia_matrix = Matrix6d::Identity() * 2;
  model.linear_damping_matrix = Matrix6d::Identity();
  model.quadratic_damping_matrix = Matrix6d::Identity() * 3;
  // Hydrostatic params default to 0, so no hydrostatic force contribution

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

TEST(Model, HydrostaticRestoringForce_ZeroAngles) {
  // When roll=0 and pitch=0, sin(0)=0, cos(0)=1
  // g(η) = [ (W-B)*0,  -(W-B)*1*0,  -(W-B)*1*1,
  //          -(yg*W-yb*B)*1 + (zg*W-zb*B)*0,
  //          (zg*W-zb*B)*0 + (xg*W-xb*B)*1,
  //          -(xg*W-xb*B)*0 - (yg*W-yb*B)*0 ]
  auv::control::Model<6> model;
  model.weight = 265.0;
  model.buoyancy = 260.0;
  model.center_of_gravity = Vector3d(0.0, 0.0, 0.02);
  model.center_of_buoyancy = Vector3d(0.0, 0.0, -0.01);

  const auto g = model.hydrostatic_restoring_force(0.0, 0.0);

  // g(0) = (W-B)*sin(0) = 0
  EXPECT_NEAR(g(0), 0.0, kEpsilon);
  // g(1) = -(W-B)*cos(0)*sin(0) = 0
  EXPECT_NEAR(g(1), 0.0, kEpsilon);
  // g(2) = -(W-B)*cos(0)*cos(0) = -(265-260) = -5
  EXPECT_NEAR(g(2), -5.0, kEpsilon);
  // g(3) = -(0*265 - 0*260)*1 + (0.02*265 - (-0.01)*260)*0 = 0
  EXPECT_NEAR(g(3), 0.0, kEpsilon);
  // g(4) = (0.02*265 - (-0.01)*260)*0 + (0*265 - 0*260)*1 = 0
  EXPECT_NEAR(g(4), 0.0, kEpsilon);
  // g(5) = 0
  EXPECT_NEAR(g(5), 0.0, kEpsilon);
}

TEST(Model, HydrostaticRestoringForce_WithPitch) {
  auv::control::Model<6> model;
  model.weight = 100.0;
  model.buoyancy = 100.0;  // neutrally buoyant
  model.center_of_gravity = Vector3d(0.0, 0.0, 0.05);
  model.center_of_buoyancy = Vector3d(0.0, 0.0, -0.05);

  const double pitch = 0.1;  // ~5.7 degrees
  const auto g = model.hydrostatic_restoring_force(0.0, pitch);

  // W == B so force components are zero
  EXPECT_NEAR(g(0), 0.0, kEpsilon);
  EXPECT_NEAR(g(1), 0.0, kEpsilon);
  EXPECT_NEAR(g(2), 0.0, kEpsilon);

  // g(3) = -(0*100 - 0*100)*cos(pitch)*cos(0) + (0.05*100 -
  // (-0.05)*100)*cos(pitch)*sin(0) = 0
  EXPECT_NEAR(g(3), 0.0, kEpsilon);

  // g(4) = (zg*W - zb*B)*sin(pitch) + (xg*W - xb*B)*cos(pitch)*cos(0)
  //       = (0.05*100 - (-0.05)*100)*sin(0.1) + 0
  //       = 10 * sin(0.1) ≈ 0.99833
  EXPECT_NEAR(g(4), 10.0 * std::sin(0.1), kEpsilon);

  // g(5) = 0
  EXPECT_NEAR(g(5), 0.0, kEpsilon);
}

TEST(Model, HydrostaticRestoringForce_WithRoll) {
  auv::control::Model<6> model;
  model.weight = 100.0;
  model.buoyancy = 100.0;
  model.center_of_gravity = Vector3d(0.0, 0.0, 0.05);
  model.center_of_buoyancy = Vector3d(0.0, 0.0, -0.05);

  const double roll = 0.2;  // ~11.5 degrees
  const auto g = model.hydrostatic_restoring_force(roll, 0.0);

  // W == B so force components are zero
  EXPECT_NEAR(g(0), 0.0, kEpsilon);
  EXPECT_NEAR(g(1), 0.0, kEpsilon);
  EXPECT_NEAR(g(2), 0.0, kEpsilon);

  // g(3) = -(0*100 - 0*100)*cos(0)*cos(roll) + (0.05*100 -
  // (-0.05)*100)*cos(0)*sin(roll)
  //       = 10 * sin(0.2) ≈ 1.98669
  EXPECT_NEAR(g(3), 10.0 * std::sin(0.2), kEpsilon);

  // g(4) = (zg*W - zb*B)*sin(0) + (xg*W - xb*B)*cos(0)*cos(roll) = 0
  EXPECT_NEAR(g(4), 0.0, kEpsilon);

  // g(5) = -(0*100 - 0*100)*cos(0)*sin(roll) - (0*100 - 0*100)*sin(0) = 0
  EXPECT_NEAR(g(5), 0.0, kEpsilon);
}

TEST(Model, HydrostaticRestoringForce_NeutralBuoyancy_ZeroCenters) {
  // When W == B and CoG == CoB, all forces and moments should be zero
  auv::control::Model<6> model;
  model.weight = 100.0;
  model.buoyancy = 100.0;
  model.center_of_gravity = Vector3d::Zero();
  model.center_of_buoyancy = Vector3d::Zero();

  const auto g = model.hydrostatic_restoring_force(0.5, 0.3);

  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(g(i), 0.0, kEpsilon);
  }
}

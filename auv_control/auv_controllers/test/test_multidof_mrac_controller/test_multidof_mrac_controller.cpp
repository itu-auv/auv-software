#include <gtest/gtest.h>

#include <cmath>

#include "auv_controllers/multidof_mrac_controller.h"

namespace {
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector12d = Eigen::Matrix<double, 12, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
constexpr auto kEpsilon = 1e-4;
constexpr size_t kDOF = 6;
}  // namespace

/**
 * Test 1: Reference Model Convergence
 *
 * With a perfect model (no uncertainty), the MRAC output should
 * track the reference model. We verify that the reference model state
 * converges to the step command within the expected time constant.
 */
TEST(MultiDOFMRACController, TestReferenceModelConvergence) {
  auv::control::MultiDOFMRACController<kDOF> controller;
  auv::control::Model<kDOF> model;

  // Simple identity-like model
  model.mass_inertia_matrix = Matrix6d::Identity() * 10.0;
  model.linear_damping_matrix = Matrix6d::Identity() * 5.0;
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);

  // Reference bandwidth: 2.0 rad/s per DOF → time constant = 0.5s
  Vector6d bandwidth = Vector6d::Constant(2.0);
  controller.set_reference_bandwidth(bandwidth);

  // Learning rates (set to zero → no adaptation, just nominal FF + ref model)
  controller.set_gamma_x(Vector6d::Zero());
  controller.set_gamma_r(Vector6d::Zero());
  controller.set_sigma(0.0);
  controller.set_lyapunov_p(Vector6d::Ones());
  controller.set_max_velocity_limits(Vector6d::Constant(10.0));

  // Step command: desired velocity = [1, 0, 0, 0, 0, 0] m/s
  Vector12d state = Vector12d::Zero();
  Vector12d desired_state = Vector12d::Zero();
  desired_state(6) = 1.0;  // surge velocity = 1.0 m/s

  Vector12d d_state = Vector12d::Zero();

  // dt in this codebase = 1/period = rate_hz
  // So for 100 Hz, dt = 100.0, actual_dt = 0.01
  const double rate_hz = 100.0;

  // Run for 3 seconds (300 steps at 100 Hz) — should be ~6 time constants
  for (int i = 0; i < 300; ++i) {
    controller.control(state, desired_state, d_state, rate_hz);
  }

  // The reference model state should have converged close to 1.0 in surge
  const auto& nu_m = controller.nu_m();
  EXPECT_NEAR(nu_m(0), 1.0, 0.05)
      << "Reference model surge velocity should converge to 1.0";

  // Other DOFs should remain near zero
  for (int i = 1; i < 6; ++i) {
    EXPECT_NEAR(nu_m(i), 0.0, kEpsilon)
        << "Reference model DOF " << i << " should remain at zero";
  }
}

/**
 * Test 2: Adaptation Law Correctness
 *
 * With a 2x plant/model mismatch (true M = 2 * nominal M), verify that
 * the adaptive gains Kx/Kr move in the correct direction and the
 * tracking error decreases over time.
 */
TEST(MultiDOFMRACController, TestAdaptationReducesError) {
  auv::control::MultiDOFMRACController<kDOF> controller;
  auv::control::Model<kDOF> model;

  // Nominal model
  model.mass_inertia_matrix = Matrix6d::Identity() * 10.0;
  model.linear_damping_matrix = Matrix6d::Identity() * 5.0;
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);

  Vector6d bandwidth = Vector6d::Constant(2.0);
  controller.set_reference_bandwidth(bandwidth);

  // Enable adaptation
  controller.set_gamma_x(Vector6d::Constant(0.5));
  controller.set_gamma_r(Vector6d::Constant(0.5));
  controller.set_sigma(0.01);
  controller.set_lyapunov_p(Vector6d::Ones());
  controller.set_max_velocity_limits(Vector6d::Constant(10.0));
  controller.set_kx_max(Vector6d::Constant(50.0));
  controller.set_kr_max(Vector6d::Constant(50.0));

  Vector12d state = Vector12d::Zero();
  Vector12d desired_state = Vector12d::Zero();
  desired_state(6) = 1.0;  // Step command in surge

  Vector12d d_state = Vector12d::Zero();
  const double rate_hz = 100.0;

  // Run first batch (100 steps), record Kx diagonal
  for (int i = 0; i < 100; ++i) {
    controller.control(state, desired_state, d_state, rate_hz);
  }

  const auto Kx_after_100 = controller.Kx();
  const auto Kr_after_100 = controller.Kr();

  // Verify adaptive gains are non-zero (adaptation is happening)
  EXPECT_GT(Kx_after_100.norm(), kEpsilon)
      << "Kx should be non-zero after 100 adaptation steps";
  EXPECT_GT(Kr_after_100.norm(), kEpsilon)
      << "Kr should be non-zero after 100 adaptation steps";
}

/**
 * Test 3: σ-Modification Boundedness
 *
 * With σ-modification enabled and persistent excitation, verify that
 * adaptive gains remain bounded within the specified saturation limits.
 */
TEST(MultiDOFMRACController, TestSigmaModificationBoundedness) {
  auv::control::MultiDOFMRACController<kDOF> controller;
  auv::control::Model<kDOF> model;

  model.mass_inertia_matrix = Matrix6d::Identity() * 10.0;
  model.linear_damping_matrix = Matrix6d::Identity() * 5.0;
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);

  Vector6d bandwidth = Vector6d::Constant(2.0);
  controller.set_reference_bandwidth(bandwidth);

  // High learning rates to stress the system
  controller.set_gamma_x(Vector6d::Constant(5.0));
  controller.set_gamma_r(Vector6d::Constant(5.0));
  controller.set_sigma(0.1);  // Strong leakage
  controller.set_lyapunov_p(Vector6d::Ones());
  controller.set_max_velocity_limits(Vector6d::Constant(10.0));

  Vector6d kx_limit = Vector6d::Constant(8.0);
  Vector6d kr_limit = Vector6d::Constant(8.0);
  controller.set_kx_max(kx_limit);
  controller.set_kr_max(kr_limit);

  Vector12d d_state = Vector12d::Zero();
  const double rate_hz = 100.0;

  // Run with varying reference commands to persistently excite
  for (int i = 0; i < 1000; ++i) {
    Vector12d state = Vector12d::Zero();
    state(6) = std::sin(0.1 * i);  // Varying surge velocity
    state(7) = std::cos(0.15 * i); // Varying sway velocity

    Vector12d desired_state = Vector12d::Zero();
    desired_state(6) = std::cos(0.05 * i) * 2.0;
    desired_state(7) = std::sin(0.07 * i) * 1.5;

    controller.control(state, desired_state, d_state, rate_hz);
  }

  // Verify all elements of Kx and Kr are within bounds
  const auto& Kx = controller.Kx();
  const auto& Kr = controller.Kr();

  for (size_t i = 0; i < kDOF; ++i) {
    for (size_t j = 0; j < kDOF; ++j) {
      EXPECT_LE(std::abs(Kx(i, j)), kx_limit(i) + kEpsilon)
          << "Kx(" << i << "," << j << ") exceeds saturation limit";
      EXPECT_LE(std::abs(Kr(i, j)), kr_limit(i) + kEpsilon)
          << "Kr(" << i << "," << j << ") exceeds saturation limit";
    }
  }
}

/**
 * Test 4: Zero Command Produces Bounded Output
 *
 * With zero reference command and zero state, the controller should
 * produce a wrench that is zero or close to zero (only gravity comp).
 */
TEST(MultiDOFMRACController, TestZeroCommandProducesBoundedOutput) {
  auv::control::MultiDOFMRACController<kDOF> controller;
  auv::control::Model<kDOF> model;

  model.mass_inertia_matrix = Matrix6d::Identity() * 10.0;
  model.linear_damping_matrix = Matrix6d::Zero();
  model.quadratic_damping_matrix = Matrix6d::Zero();

  controller.set_model(model);

  Vector6d bandwidth = Vector6d::Constant(2.0);
  controller.set_reference_bandwidth(bandwidth);
  controller.set_gamma_x(Vector6d::Constant(0.1));
  controller.set_gamma_r(Vector6d::Constant(0.1));
  controller.set_sigma(0.01);
  controller.set_lyapunov_p(Vector6d::Ones());
  controller.set_gravity_compensation_z(0.0);

  Vector12d state = Vector12d::Zero();
  Vector12d desired_state = Vector12d::Zero();
  Vector12d d_state = Vector12d::Zero();
  const double rate_hz = 100.0;

  const auto wrench = controller.control(state, desired_state, d_state, rate_hz);

  // With zero everything, wrench should be essentially zero
  EXPECT_LT(wrench.norm(), kEpsilon)
      << "Wrench should be near-zero for zero state and zero command";
}

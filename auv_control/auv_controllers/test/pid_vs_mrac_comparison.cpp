/**
 * @file pid_vs_mrac_comparison.cpp
 * @brief Closed-loop comparison of PID and MRAC controllers under normal
 *        and damaged vehicle dynamics.
 *
 * Simulates a 6-DOF AUV plant: M·ν̇ = τ − D_l·ν − D_q·|ν|⊙ν
 * and runs both controllers through step responses.
 *
 * Output: CSV files for each scenario to be plotted with plot_comparison.py
 */

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "auv_controllers/multidof_mrac_controller.h"
#include "auv_controllers/multidof_pid_controller.h"

// ─── Type aliases ────────────────────────────────────────────────────────────
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;

// ─── CSV data row ────────────────────────────────────────────────────────────
struct DataRow {
  double time;
  Vec6 desired_vel;
  Vec6 actual_vel;
  Vec6 wrench;
  Vec6 error;
  Vec6 ref_model_vel;   // Only meaningful for MRAC
  Vec6 kx_diag;         // Only meaningful for MRAC
};

// ─── Build the real AUV model from default.yaml ──────────────────────────────
auv::control::Model<6> make_nominal_model() {
  auv::control::Model<6> m;
  m.mass_inertia_matrix = Mat6::Zero();
  m.mass_inertia_matrix.diagonal() << 27.0, 27.0, 27.0, 0.75, 2.245, 2.116;

  m.linear_damping_matrix = Mat6::Zero();
  m.linear_damping_matrix.diagonal() << 30.1481, 31.4748, 89.2, 0.0, 0.0,
      1.87816;

  m.quadratic_damping_matrix = Mat6::Zero();
  m.quadratic_damping_matrix.diagonal() << 77.0139, 155.237, 136.16, 0.0, 0.0,
      7.98053;

  return m;
}

// ─── Damaged model: applied at t=5s ──────────────────────────────────────────
// Simulates flooding (mass +50%), fin damage (asymmetric damping), and
// cross-coupling (off-diagonal damping terms).
auv::control::Model<6> make_damaged_model() {
  auto m = make_nominal_model();

  // Mass increase: 50% heavier (flooding / payload shift)
  m.mass_inertia_matrix *= 1.5;

  // Asymmetric damping increase
  m.linear_damping_matrix(0, 0) *= 2.5;   // surge damping way up (damaged fin)
  m.linear_damping_matrix(1, 1) *= 1.8;   // sway slightly up
  m.linear_damping_matrix(2, 2) *= 1.3;   // heave slightly up

  m.quadratic_damping_matrix(0, 0) *= 2.0;
  m.quadratic_damping_matrix(1, 1) *= 3.0;  // big sway quad. damping

  // Cross-coupling: surge creates yaw torque (asymmetric drag)
  m.linear_damping_matrix(5, 0) = 5.0;   // surge velocity -> yaw torque
  m.linear_damping_matrix(0, 5) = 3.0;   // yaw rate -> surge drag

  return m;
}

// ─── Simple forward-Euler plant ──────────────────────────────────────────────
struct Plant {
  Vec6 position = Vec6::Zero();
  Vec6 velocity = Vec6::Zero();

  void step(const Vec6& wrench, const auv::control::Model<6>& model,
            double dt) {
    // M·ν̇ = τ − D_l·ν − D_q·|ν|⊙ν
    Vec6 damping = model.linear_damping_matrix * velocity +
                   model.quadratic_damping_matrix *
                       velocity.cwiseAbs().cwiseProduct(velocity);
    Vec6 accel =
        model.mass_inertia_matrix.inverse() * (wrench - damping);
    velocity += accel * dt;
    position += velocity * dt;
  }

  Vec12 state() const {
    Vec12 s;
    s << position, velocity;
    return s;
  }

  Vec12 d_state() const {
    Vec12 ds;
    ds << velocity, Vec6::Zero();  // acceleration filled as zero (controller doesn't use it heavily)
    return ds;
  }
};

// ─── Configure the PID controller (from default.yaml) ────────────────────────
void configure_pid(auv::control::MultiDOFPIDController<6>& pid,
                   const auv::control::Model<6>& model) {
  pid.set_model(model);

  Eigen::Matrix<double, 12, 1> kp, ki, kd, integral_clamp;
  kp << 1.0, 1.0, 0.8, -1.0, -0.5, 2.0,
        1.0, 1.0, 1.0, -10.0, -10.0, 6.0;
  ki << 0.0, 0.0, 0.0003, -0.003, -0.003, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0002;
  kd << 0.7, 0.7, 1.2, -0.1, -0.007, 0.3,
        1.0, 1.0, 1.0, -1.0, -1.0, 10.0;
  integral_clamp << 0.0, 0.0, 100.0, 100.0, 100.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 100.0;

  pid.set_kp(kp);
  pid.set_ki(ki);
  pid.set_kd(kd);
  pid.set_integral_clamp_limits(integral_clamp);

  Vec6 max_vel;
  max_vel << 0.6, 0.6, 0.6, 0.4, 0.4, 0.4;
  pid.set_max_velocity_limits(max_vel);
  pid.set_gravity_compensation_z(0.0);
}

// ─── Configure the MRAC controller (from default_mrac.yaml) ─────────────────
void configure_mrac(auv::control::MultiDOFMRACController<6>& mrac,
                    const auv::control::Model<6>& model) {
  mrac.set_model(model);

  Vec6 bandwidth;
  bandwidth << 1.0, 1.0, 1.5, 1.5, 1.5, 1.5;
  mrac.set_reference_bandwidth(bandwidth);

  Vec6 gamma_x, gamma_r;
  gamma_x << 0.5, 0.5, 1.5, 0.2, 0.2, 0.2;
  gamma_r << 0.5, 0.5, 1.5, 0.2, 0.2, 0.2;
  mrac.set_gamma_x(gamma_x);
  mrac.set_gamma_r(gamma_r);
  mrac.set_sigma(0.01);

  Vec6 lyapunov_p;
  lyapunov_p << 1.0, 1.0, 2.0, 1.0, 1.0, 1.0;
  mrac.set_lyapunov_p(lyapunov_p);

  Vec6 kx_max, kr_max;
  kx_max << 10.0, 10.0, 10.0, 5.0, 5.0, 5.0;
  kr_max << 10.0, 10.0, 10.0, 5.0, 5.0, 5.0;
  mrac.set_kx_max(kx_max);
  mrac.set_kr_max(kr_max);

  // Position outer loop gains
  Vec6 pos_kp, pos_ki, pos_kd, pos_clamp;
  pos_kp << 1.0, 1.0, 0.8, 0.0, 0.0, 2.0;
  pos_ki << 0.0, 0.0, 0.001, 0.0, 0.0, 0.0;
  pos_kd << 1.0, 1.0, 0.1, 0.0, 0.0, 0.3;
  pos_clamp << 0.0, 0.0, 5.0, 0.0, 0.0, 0.0;
  mrac.set_kp_position(pos_kp);
  mrac.set_ki_position(pos_ki);
  mrac.set_kd_position(pos_kd);
  mrac.set_integral_clamp_limits(pos_clamp);

  Vec6 max_vel;
  max_vel << 0.6, 0.6, 0.6, 0.4, 0.4, 0.4;
  mrac.set_max_velocity_limits(max_vel);
  mrac.set_gravity_compensation_z(0.0);
}

// ─── Write CSV ───────────────────────────────────────────────────────────────
void write_csv(const std::string& filename, const std::vector<DataRow>& data,
               bool is_mrac) {
  std::ofstream f(filename);
  if (!f.is_open()) {
    std::cerr << "ERROR: Could not open " << filename << std::endl;
    return;
  }

  f << "time,des_surge,act_surge,des_sway,act_sway,des_heave,act_heave,"
       "des_roll,act_roll,des_pitch,act_pitch,des_yaw,act_yaw,"
       "wrench_surge,wrench_sway,wrench_heave,wrench_roll,wrench_pitch,"
       "wrench_yaw,err_surge,err_sway,err_heave,err_roll,err_pitch,err_yaw";
  if (is_mrac) {
    f << ",ref_surge,ref_sway,ref_heave,ref_roll,ref_pitch,ref_yaw,"
         "kx_surge,kx_sway,kx_heave,kx_roll,kx_pitch,kx_yaw";
  }
  f << "\n";

  for (const auto& row : data) {
    f << row.time;
    for (int i = 0; i < 6; ++i)
      f << "," << row.desired_vel(i) << "," << row.actual_vel(i);
    for (int i = 0; i < 6; ++i) f << "," << row.wrench(i);
    for (int i = 0; i < 6; ++i) f << "," << row.error(i);
    if (is_mrac) {
      for (int i = 0; i < 6; ++i) f << "," << row.ref_model_vel(i);
      for (int i = 0; i < 6; ++i) f << "," << row.kx_diag(i);
    }
    f << "\n";
  }

  f.close();
  std::cout << "  Wrote " << filename << " (" << data.size() << " rows)\n";
}

// ─── Run a PID simulation scenario ──────────────────────────────────────────
std::vector<DataRow> run_pid_scenario(
    const auv::control::Model<6>& controller_model,
    const auv::control::Model<6>& plant_model_normal,
    const auv::control::Model<6>& plant_model_damaged,
    const Vec6& desired_velocity, double sim_duration, double rate_hz,
    double damage_time) {
  auv::control::MultiDOFPIDController<6> pid;
  configure_pid(pid, controller_model);

  Plant plant;
  const double actual_dt = 1.0 / rate_hz;
  const int steps = static_cast<int>(sim_duration * rate_hz);
  std::vector<DataRow> data;
  data.reserve(steps);

  for (int step = 0; step < steps; ++step) {
    double t = step * actual_dt;

    // Set desired_position = current position so the position
    // outer loop produces zero output → pure velocity tracking test.
    Vec12 desired_state;
    desired_state << plant.position, desired_velocity;

    // Select plant model based on damage time
    const auto& current_plant_model =
        (t >= damage_time) ? plant_model_damaged : plant_model_normal;

    Vec12 state = plant.state();
    Vec12 d_state = plant.d_state();

    Vec6 wrench = pid.control(state, desired_state, d_state, rate_hz);
    plant.step(wrench, current_plant_model, actual_dt);

    // Record data
    DataRow row;
    row.time = t;
    row.desired_vel = desired_velocity;
    row.actual_vel = plant.velocity;
    row.wrench = wrench;
    row.error = desired_velocity - plant.velocity;
    row.ref_model_vel = Vec6::Zero();
    row.kx_diag = Vec6::Zero();
    data.push_back(row);
  }

  return data;
}

// ─── Run an MRAC simulation scenario ─────────────────────────────────────────
std::vector<DataRow> run_mrac_scenario(
    const auv::control::Model<6>& controller_model,
    const auv::control::Model<6>& plant_model_normal,
    const auv::control::Model<6>& plant_model_damaged,
    const Vec6& desired_velocity, double sim_duration, double rate_hz,
    double damage_time) {
  auv::control::MultiDOFMRACController<6> mrac;
  configure_mrac(mrac, controller_model);

  Plant plant;
  const double actual_dt = 1.0 / rate_hz;
  const int steps = static_cast<int>(sim_duration * rate_hz);
  std::vector<DataRow> data;
  data.reserve(steps);

  for (int step = 0; step < steps; ++step) {
    double t = step * actual_dt;

    // Set desired_position = current position so the position
    // outer loop produces zero output → pure velocity tracking test.
    Vec12 desired_state;
    desired_state << plant.position, desired_velocity;

    const auto& current_plant_model =
        (t >= damage_time) ? plant_model_damaged : plant_model_normal;

    Vec12 state = plant.state();
    Vec12 d_state = plant.d_state();

    Vec6 wrench = mrac.control(state, desired_state, d_state, rate_hz);
    plant.step(wrench, current_plant_model, actual_dt);

    // Record data
    DataRow row;
    row.time = t;
    row.desired_vel = desired_velocity;
    row.actual_vel = plant.velocity;
    row.wrench = wrench;
    row.error = desired_velocity - plant.velocity;
    row.ref_model_vel = mrac.nu_m();
    // Extract Kx diagonal
    const auto& Kx = mrac.Kx();
    for (int i = 0; i < 6; ++i) row.kx_diag(i) = Kx(i, i);
    data.push_back(row);
  }

  return data;
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
  std::cout << "═══════════════════════════════════════════════════════════\n";
  std::cout << "  PID vs MRAC Adaptability Comparison\n";
  std::cout << "═══════════════════════════════════════════════════════════\n\n";

  // ── Models ──
  auto nominal_model = make_nominal_model();
  auto damaged_model = make_damaged_model();

  // ── Desired velocity step command ──
  Vec6 desired_velocity = Vec6::Zero();
  desired_velocity(0) = 0.5;    // surge = 0.5 m/s
  desired_velocity(2) = -0.3;   // heave = -0.3 m/s (dive)
  desired_velocity(5) = 0.3;    // yaw rate = 0.3 rad/s

  const double sim_duration = 20.0;  // seconds
  const double rate_hz = 100.0;      // Hz
  const double damage_time = 5.0;    // seconds

  // Determine output directory (same as this source file)
  std::string output_dir = ".";
  if (argc > 1) {
    output_dir = argv[1];
  }

  // ── Scenario 1: PID + Normal ──
  std::cout << "[1/4] Running PID + Normal dynamics...\n";
  auto pid_normal = run_pid_scenario(
      nominal_model, nominal_model, nominal_model,  // no damage
      desired_velocity, sim_duration, rate_hz,
      sim_duration + 1.0);  // damage_time > sim_duration = never
  write_csv(output_dir + "/pid_normal.csv", pid_normal, false);

  // ── Scenario 2: MRAC + Normal ──
  std::cout << "[2/4] Running MRAC + Normal dynamics...\n";
  auto mrac_normal = run_mrac_scenario(
      nominal_model, nominal_model, nominal_model,
      desired_velocity, sim_duration, rate_hz, sim_duration + 1.0);
  write_csv(output_dir + "/mrac_normal.csv", mrac_normal, true);

  // ── Scenario 3: PID + Damaged ──
  std::cout << "[3/4] Running PID + Damaged dynamics (damage at t="
            << damage_time << "s)...\n";
  auto pid_damaged = run_pid_scenario(
      nominal_model, nominal_model, damaged_model,
      desired_velocity, sim_duration, rate_hz, damage_time);
  write_csv(output_dir + "/pid_damaged.csv", pid_damaged, false);

  // ── Scenario 4: MRAC + Damaged ──
  std::cout << "[4/4] Running MRAC + Damaged dynamics (damage at t="
            << damage_time << "s)...\n";
  auto mrac_damaged = run_mrac_scenario(
      nominal_model, nominal_model, damaged_model,
      desired_velocity, sim_duration, rate_hz, damage_time);
  write_csv(output_dir + "/mrac_damaged.csv", mrac_damaged, true);

  std::cout << "\n✓ All scenarios complete. Run plot_comparison.py to visualize.\n";
  return 0;
}

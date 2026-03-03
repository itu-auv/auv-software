#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <vector>

#include "angles/angles.h"
#include "auv_controllers/controller_base.h"

namespace auv {
namespace control {

/**
 * @brief Model Reference Adaptive Controller (MRAC) for multi-DOF systems.
 *
 * Replaces the inner velocity PID loop of the cascaded control architecture.
 * Uses Lyapunov-based adaptation laws to compensate for parametric uncertainty
 * in the hydrodynamic model (M, D matrices).
 *
 * Control Law:
 *   τ = M̂(A_m ν_m + B_m r + K_x ν + K_r r) + D̂_l ν + D̂_q |ν|⊙ν + g_comp
 *
 * Adaptation Laws (with σ-modification):
 *   K̇_x = -Γ_x (ν eᵀ P B_m) - σ Γ_x K_x
 *   K̇_r = -Γ_r (r eᵀ P B_m) - σ Γ_r K_r
 *
 * Reference Model:
 *   ν̇_m = A_m ν_m + B_m r
 *
 * Template parameter N = number of DOFs (6 for standard AUV).
 *
 * Based on: Fossen, T.I. "Handbook of Marine Craft Hydrodynamics and Motion
 * Control", Ch. 11-12.
 */
template <size_t N>
class MultiDOFMRACController : public ControllerBase<N> {
  static_assert(N >= 6,
    "MultiDOFMRACController requires at least 6 DOF; "
    "yes.");

  using Base = ControllerBase<N>;
  using WrenchVector = typename Base::WrenchVector;
  using StateVector = typename Base::StateVector;
  using Vectornd = Eigen::Matrix<double, N, 1>;
  using Matrixnd = Eigen::Matrix<double, N, N>;

 public:
  MultiDOFMRACController()
      : Am_(Matrixnd::Zero()),
        Bm_(Matrixnd::Zero()),
        P_(Matrixnd::Identity()),
        gamma_x_(Matrixnd::Identity()),
        gamma_r_(Matrixnd::Identity()),
        Kx_(Matrixnd::Zero()),
        Kr_(Matrixnd::Zero()),
        nu_m_(Vectornd::Zero()),
        sigma_(0.01),
        gravity_compensation_z_(0.0),
        kx_max_(Vectornd::Constant(10.0)),
        kr_max_(Vectornd::Constant(10.0)),
        max_velocity_limits_(Vectornd::Constant(1e6)),
        initialized_(false) {}

  // ──────────────────── Setters (called from ROS layer) ────────────────────

  /**
   * @brief Set the reference model matrices from bandwidth vector.
   *
   * For a first-order ideal response per DOF:
   *   A_m = diag(-ω_1, ..., -ω_N)
   *   B_m = diag( ω_1, ...,  ω_N)   (unity DC gain)
   *
   * @param bandwidth Vector of desired bandwidths [rad/s] per DOF
   */
  void set_reference_bandwidth(const Vectornd& bandwidth) {
    Am_ = (-bandwidth).asDiagonal();
    Bm_ = bandwidth.asDiagonal();
  }

  /**
   * @brief Set the reference model matrices directly.
   * Use this if you need non-diagonal or coupled reference dynamics.
   */
  void set_reference_model(const Matrixnd& Am, const Matrixnd& Bm) {
    Am_ = Am;
    Bm_ = Bm;
  }

  /** @brief Set the state-feedback adaptation learning rate (diagonal). */
  void set_gamma_x(const Vectornd& gamma_x) {
    gamma_x_ = gamma_x.asDiagonal();
  }

  /** @brief Set the input-feedforward adaptation learning rate (diagonal). */
  void set_gamma_r(const Vectornd& gamma_r) {
    gamma_r_ = gamma_r.asDiagonal();
  }

  /** @brief Set the Lyapunov P matrix (diagonal). */
  void set_lyapunov_p(const Vectornd& p_diag) {
    // P must be symmetric positive definite for Lyapunov stability guarantee.
    // Using diagonal form ensures SPD as long as all entries are positive.
    for (size_t i = 0; i < N; ++i) {
      assert(p_diag(i) > 0.0 &&
        "All Lyapunov P diagonal entries must be strictly positive.");
    }
    P_ = p_diag.asDiagonal();
  }

  /** @brief Set the σ-modification leakage factor. */
  void set_sigma(double sigma) {
    assert(sigma >= 0.0 &&
      "sigma (σ-modification leakage) must be non-negative.");
    sigma_ = sigma;
  }

  /** @brief Set the adaptive gain saturation limits for Kx (per-element). */
  void set_kx_max(const Vectornd& kx_max) { kx_max_ = kx_max; }

  /** @brief Set the adaptive gain saturation limits for Kr (per-element). */
  void set_kr_max(const Vectornd& kr_max) { kr_max_ = kr_max; }

  /** @brief Set the gravity compensation for z-axis. */
  void set_gravity_compensation_z(double compensation) {
    gravity_compensation_z_ = compensation;
  }

  /** @brief Set the max velocity limits for the position controller output. */
  void set_max_velocity_limits(const Vectornd& limits) {
    max_velocity_limits_ = limits;
  }

  /** @brief Set the position PID gains (outer loop, kept from existing PID). */
  void set_kp_position(const Vectornd& kp) { kp_pos_ = kp.asDiagonal(); }
  void set_ki_position(const Vectornd& ki) { ki_pos_ = ki.asDiagonal(); }
  void set_kd_position(const Vectornd& kd) { kd_pos_ = kd.asDiagonal(); }

  /** @brief Set integral clamp limits for position outer loop anti-windup. */
  void set_integral_clamp_limits(const Vectornd& limits) {
    integral_clamp_limits_ = limits;
  }

  /** @brief Reset the adaptive gains and reference model state to zero. */
  void reset_adaptation() {
    Kx_ = Matrixnd::Zero();
    Kr_ = Matrixnd::Zero();
    nu_m_ = Vectornd::Zero();
    pos_integral_ = Vectornd::Zero();
    initialized_ = false;
  }

  // ──────────────────── Read-only accessors (diagnostics) ──────────────────

  const Matrixnd& Kx() const { return Kx_; }
  const Matrixnd& Kr() const { return Kr_; }
  const Vectornd& nu_m() const { return nu_m_; }

  /**
   * @brief Calculate the MRAC control output wrench.
   *
   * @param state        Current state [position(6); velocity(6)]
   * @param desired_state Desired state [position(6); velocity(6)]
   * @param d_state      State derivative [velocity(6); acceleration(6)]
   * @param dt           Delta time (NOTE: in this codebase dt = 1/period,
   *                     i.e. the rate in Hz. We compute actual_dt = 1/dt.)
   * @return WrenchVector Body-frame force/torque output
   */
  WrenchVector control(const StateVector& state,
                       const StateVector& desired_state,
                       const StateVector& d_state, const double dt) {
    // ── Compute actual timestep ──
    // The codebase passes dt = 1/T (Hz), we need actual period T
    const double actual_dt = 1.0 / dt;

    // Sanity-check: dt is expected in Hz (e.g. 10–200 Hz typical for AUV)
    assert(dt >= 1.0 && dt <= 10000.0 &&
      "dt must be the controller rate in Hz, not a period in seconds.");

    // ── Position PID outer loop (retained from existing architecture melih and talha wrote
    // Zoktay and hilal) ──
    const auto inverse_rotation_matrix = get_world_to_body_rotation(state);

    Vectornd pos_pid_output = Vectornd::Zero();
    {
      const auto position_state = state.template head<N>();
      const auto desired_position = desired_state.template head<N>();
      const auto velocity_state = d_state.template head<N>();

      Vectornd error = Vectornd::Zero();
      error.template head<3>() =
          desired_position.template head<3>() -
          position_state.template head<3>();
      error.template head<3>() =
          inverse_rotation_matrix * error.template head<3>();
      for (size_t i = 3; i < N; ++i) {
        error(i) = angles::shortest_angular_distance(position_state(i),
                                                     desired_position(i));
      }

      const auto p_term = kp_pos_ * error;

      pos_integral_ += error * actual_dt;

      // Anti-windup clamping
      for (size_t i = 0; i < N; ++i) {
        const double limit = integral_clamp_limits_(i);
        if (limit > 0) {
          pos_integral_(i) =
              std::max(-limit, std::min(limit, pos_integral_(i)));
        }
      }

      const auto i_term = ki_pos_ * pos_integral_;
      const auto d_term =
          kd_pos_ * (Vectornd::Zero() - velocity_state);

      pos_pid_output = p_term + i_term + d_term;
    }

    // ── Construct reference command r(t) ──
    // r = desired_velocity (from cmd_vel) + position PID correction, clamped
    const Vectornd r = (desired_state.template tail<N>() + pos_pid_output)
                           .cwiseMin(max_velocity_limits_)
                           .cwiseMax(-max_velocity_limits_);

    // ── Current body-frame velocity ──
    const Vectornd nu = state.template tail<N>();

    // ── Initialize reference model state on first call ──
    if (!initialized_) {
      nu_m_ = nu;
      initialized_ = true;
    }

    // ── Integrate reference model: exact ZOH for diagonal Am_, Bm_ ──
    // For DOF i: ν_m_i(k+1) = exp(a_i·T)·ν_m_i(k) + (exp(a_i·T)−1)/a_i·b_i·r_i
    // Degenerates to Euler if Am_ is not diagonal (fallback provided).
    {
      const Vectornd a = Am_.diagonal();  // a_i = -ω_i (negative)
      const Vectornd b = Bm_.diagonal();
      for (size_t i = 0; i < N; ++i) {
        const double ai = a(i);
        const double bi = b(i);
        if (std::abs(ai) < 1e-10) {
          // Near-zero pole: use Euler
          nu_m_(i) += (ai * nu_m_(i) + bi * r(i)) * actual_dt;
        } else {
          const double eaT = std::exp(ai * actual_dt);
          nu_m_(i) = eaT * nu_m_(i) + (eaT - 1.0) / ai * bi * r(i);
        }
      }
    }

    // ── Tracking error ──
    const Vectornd e = nu - nu_m_;

    // ── Nominal feedforward using existing Model struct ──
    // τ_ff = D_l · ν + D_q · |ν| ⊙ ν
    const Vectornd damping_ff =
        this->model().linear_damping_matrix * nu +
        this->model().quadratic_damping_matrix *
            nu.cwiseAbs().cwiseProduct(nu);

    // ── MRAC control law ──
    // τ = M̂·(A_m ν_m + B_m r + K_x ν + K_r r) + damping_ff + g_comp
    // Note: A_m acts on ν_m (reference model state), not on plant state ν.
    const Vectornd mrac_accel =
        Am_ * nu_m_ + Bm_ * r + Kx_ * nu + Kr_ * r;

    WrenchVector wrench =
        this->model().mass_inertia_matrix * mrac_accel + damping_ff;

    // ── Gravity compensation ──
    Eigen::Vector3d gravity_force_global = Eigen::Vector3d::Zero();
    gravity_force_global(2) = gravity_compensation_z_;
    wrench.template head<3>() +=
        inverse_rotation_matrix * gravity_force_global;

    // ── Adaptation law update (with σ-modification) ──
    // eᵀ P B_m → row vector (1×N) × (N×N) × (N×N) = (1×N) → transpose = N×1
    const Vectornd e_T_P_Bm = (e.transpose() * P_ * Bm_).transpose();

    // K̇_x = -Γ_x (ν · eᵀPBmᵀ) - σ Γ_x K_x
    //      ν is N×1, eᵀPBm is N×1, outer product = N×N
    Kx_ += (-gamma_x_ * (nu * e_T_P_Bm.transpose()) -
             sigma_ * gamma_x_ * Kx_) *
            actual_dt;

    // K̇_r = -Γ_r (r · eᵀPBmᵀ) - σ Γ_r K_r
    Kr_ += (-gamma_r_ * (r * e_T_P_Bm.transpose()) -
             sigma_ * gamma_r_ * Kr_) *
            actual_dt;

    // ── Gain saturation (element-wise clamping) ──
    clamp_matrix(Kx_, kx_max_);
    clamp_matrix(Kr_, kr_max_);

    return wrench;
  }

 private:
  /**
   * @brief Clamp each element of a matrix to [-limit, +limit] per row.
   * limit(i) applies to all elements in row i.
   */
  void clamp_matrix(Matrixnd& mat, const Vectornd& limits) const {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        mat(i, j) = std::max(-limits(i), std::min(limits(i), mat(i, j)));
      }
    }
  }

  /**
   * @brief Compute world-to-body rotation matrix from Euler angles in state.
   */
  Eigen::Matrix3d get_world_to_body_rotation(const StateVector& state) const {
    Eigen::AngleAxisd roll_angle(state[3], Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch_angle(state[4], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw_angle(state[5], Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond rotation = yaw_angle * pitch_angle * roll_angle;
    return rotation.matrix().transpose();  // body->world transposed = world->body
  }

  // ── Reference model ──
  Matrixnd Am_;       ///< Reference model state matrix (Hurwitz, typically diagonal)
  Matrixnd Bm_;       ///< Reference model input matrix
  Vectornd nu_m_;     ///< Reference model velocity state

  // ── Adaptation parameters ──
  Matrixnd P_;        ///< Lyapunov matrix (symmetric positive definite)
  Matrixnd gamma_x_;  ///< Learning rate for state feedback gains
  Matrixnd gamma_r_;  ///< Learning rate for input feedforward gains
  double sigma_;      ///< σ-modification leakage factor

  // ── Adaptive gains (updated online) ──
  Matrixnd Kx_;       ///< State feedback adaptive gain matrix
  Matrixnd Kr_;       ///< Input feedforward adaptive gain matrix

  // ── Gain saturation limits ──
  Vectornd kx_max_;   ///< Max absolute value per row for Kx elements
  Vectornd kr_max_;   ///< Max absolute value per row for Kr elements

  // ── Position PID outer loop (retained from existing architecture) ──
  Matrixnd kp_pos_{Matrixnd::Zero()};
  Matrixnd ki_pos_{Matrixnd::Zero()};
  Matrixnd kd_pos_{Matrixnd::Zero()};
  Vectornd pos_integral_{Vectornd::Zero()};
  /// Anti-windup clamp per DOF. Zero (default) disables clamping for that DOF.
  /// Set via set_integral_clamp_limits() before use to enable anti-windup.
  Vectornd integral_clamp_limits_{Vectornd::Zero()};

  // ── Misc ──
  double gravity_compensation_z_;
  Vectornd max_velocity_limits_;
  bool initialized_;
};

using SixDOFMRACController = MultiDOFMRACController<6>;

}  // namespace control
}  // namespace auv

#pragma once
#include <Eigen/Core>
#include <cmath>
#include <ostream>

namespace auv {
namespace control {

template <size_t N>
struct Model {
  using Matrix = Eigen::Matrix<double, N, N>;
  using Vector = Eigen::Matrix<double, N, 1>;
  using Vector3 = Eigen::Matrix<double, 3, 1>;
  using Matrix3 = Eigen::Matrix<double, 3, 3>;

  Model() = default;

  // copy constructor
  Model(const Model& other) {
    mass_inertia_matrix = other.mass_inertia_matrix;
    linear_damping_matrix = other.linear_damping_matrix;
    quadratic_damping_matrix = other.quadratic_damping_matrix;
    weight = other.weight;
    buoyancy = other.buoyancy;
    center_of_gravity = other.center_of_gravity;
    center_of_buoyancy = other.center_of_buoyancy;
  }

  // copy assignment
  Model& operator=(const Model& other) {
    if (this == &other) {
      return *this;
    }

    mass_inertia_matrix = other.mass_inertia_matrix;
    linear_damping_matrix = other.linear_damping_matrix;
    quadratic_damping_matrix = other.quadratic_damping_matrix;
    weight = other.weight;
    buoyancy = other.buoyancy;
    center_of_gravity = other.center_of_gravity;
    center_of_buoyancy = other.center_of_buoyancy;

    return *this;
  }

  // move constructor
  Model(Model&& other) noexcept {
    mass_inertia_matrix = std::move(other.mass_inertia_matrix);
    linear_damping_matrix = std::move(other.linear_damping_matrix);
    quadratic_damping_matrix = std::move(other.quadratic_damping_matrix);
    weight = other.weight;
    buoyancy = other.buoyancy;
    center_of_gravity = std::move(other.center_of_gravity);
    center_of_buoyancy = std::move(other.center_of_buoyancy);
  }

  /**
   * @brief Compute the Coriolis-Centrifugal matrix C(v) for a 6-DOF system.
   *
   * The Coriolis matrix is computed from the mass-inertia matrix M:
   * M = [M11  M12]   where M11 = m*I + m*S(r_g)^T*S(r_g)
   *     [M21  M22]         M12 = -m*S(r_g)
   *                        M21 = m*S(r_g)
   *                        M22 = I_o (inertia tensor about origin)
   *
   * C(v) is computed using the parameterization that ensures C(v) is
   * skew-symmetric: C(v) + C(v)^T = dM/dt (which is 0 for constant M)
   *
   * For rigid body:
   * C_RB(v) = [  0          -S(M11*v1 + M12*v2) ]
   *           [ -S(M11*v1 + M12*v2)  -S(M21*v1 + M22*v2) ]
   *
   * where v = [v1; v2] = [linear_vel; angular_vel]
   * and S(.) is the skew-symmetric matrix operator
   *
   * @param velocity The 6-DOF velocity vector [u, v, w, p, q, r]
   * @return The 6x6 Coriolis-Centrifugal matrix
   */
  Matrix coriolis_matrix(const Vector& velocity) const {
    static_assert(N == 6, "Coriolis matrix is only defined for 6-DOF systems");

    Matrix C = Matrix::Zero();

    // Extract sub-matrices from mass-inertia matrix
    // M = [M11  M12]
    //     [M21  M22]
    const Matrix3 M11 = mass_inertia_matrix.template block<3, 3>(0, 0);
    const Matrix3 M12 = mass_inertia_matrix.template block<3, 3>(0, 3);
    const Matrix3 M21 = mass_inertia_matrix.template block<3, 3>(3, 0);
    const Matrix3 M22 = mass_inertia_matrix.template block<3, 3>(3, 3);

    // Extract linear and angular velocities
    const Vector3 v1 =
        velocity.template head<3>();  // linear velocity [u, v, w]
    const Vector3 v2 =
        velocity.template tail<3>();  // angular velocity [p, q, r]

    // Compute the momentum-like terms
    const Vector3 Mv1 = M11 * v1 + M12 * v2;
    const Vector3 Mv2 = M21 * v1 + M22 * v2;

    // C(v) = [     0        -S(Mv1) ]
    //        [  -S(Mv1)     -S(Mv2) ]
    C.template block<3, 3>(0, 3) = -skew(Mv1);
    C.template block<3, 3>(3, 0) = -skew(Mv1);
    C.template block<3, 3>(3, 3) = -skew(Mv2);

    return C;
  }

  /**
   * @brief Compute the hydrostatic restoring force/moment vector g(η).
   *
   * Uses the Fossen formulation for underwater vehicles:
   * g(η) = [ (W - B) sin(θ) ] [ -(W - B) cos(θ) sin(φ) ] [ -(W - B) cos(θ)
   * cos(φ)                                                 ] [ -(y_g*W - y_b*B)
   * cos(θ)cos(φ) + (z_g*W - z_b*B) cos(θ)sin(φ)          ] [  (z_g*W - z_b*B)
   * sin(θ) + (x_g*W - x_b*B) cos(θ)cos(φ)                ] [ -(x_g*W - x_b*B)
   * cos(θ)sin(φ) - (y_g*W - y_b*B) sin(θ)                ]
   *
   * where W = weight, B = buoyancy, r_g = [x_g, y_g, z_g] (CoG),
   *       r_b = [x_b, y_b, z_b] (CoB), φ = roll, θ = pitch
   *
   * @param roll  Roll angle (φ) in radians
   * @param pitch Pitch angle (θ) in radians
   * @return The 6-DOF hydrostatic restoring force vector (in body frame)
   */
  Vector hydrostatic_restoring_force(double roll, double pitch) const {
    static_assert(N == 6,
                  "Hydrostatic restoring force is only defined for 6-DOF "
                  "systems");

    const double W = weight;
    const double B = buoyancy;
    const double xg = center_of_gravity(0);
    const double yg = center_of_gravity(1);
    const double zg = center_of_gravity(2);
    const double xb = center_of_buoyancy(0);
    const double yb = center_of_buoyancy(1);
    const double zb = center_of_buoyancy(2);

    const double sp = std::sin(pitch);  // sin(θ)
    const double cp = std::cos(pitch);  // cos(θ)
    const double sr = std::sin(roll);   // sin(φ)
    const double cr = std::cos(roll);   // cos(φ)

    Vector g = Vector::Zero();
    g(0) = (W - B) * sp;
    g(1) = -(W - B) * cp * sr;
    g(2) = -(W - B) * cp * cr;
    g(3) = -(yg * W - yb * B) * cp * cr + (zg * W - zb * B) * cp * sr;
    g(4) = (zg * W - zb * B) * sp + (xg * W - xb * B) * cp * cr;
    g(5) = -(xg * W - xb * B) * cp * sr - (yg * W - yb * B) * sp;

    return g;
  }

  Matrix mass_inertia_matrix;
  Matrix linear_damping_matrix;
  Matrix quadratic_damping_matrix;

  // Hydrostatic parameters
  double weight{0.0};                           // W = m*g (N)
  double buoyancy{0.0};                         // B (N)
  Vector3 center_of_gravity{Vector3::Zero()};   // CoG in body frame [m]
  Vector3 center_of_buoyancy{Vector3::Zero()};  // CoB in body frame [m]

 private:
  /**
   * @brief Compute the skew-symmetric matrix S(v) such that S(v)*u = v x u
   *
   * For v = [v1, v2, v3]^T:
   * S(v) = [  0   -v3   v2 ]
   *        [  v3   0   -v1 ]
   *        [ -v2   v1   0  ]
   *
   * @param v A 3D vector
   * @return The 3x3 skew-symmetric matrix
   */
  static Matrix3 skew(const Vector3& v) {
    Matrix3 S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
  }
};

template <size_t N>
inline std::ostream& operator<<(std::ostream& os, const Model<N>& model) {
  os << "mass_inertia_matrix: " << std::endl
     << model.mass_inertia_matrix << std::endl;
  os << "linear_damping_matrix: " << std::endl
     << model.linear_damping_matrix << std::endl;
  os << "quadratic_damping_matrix: " << std::endl
     << model.quadratic_damping_matrix << std::endl;
  os << "weight: " << model.weight << std::endl;
  os << "buoyancy: " << model.buoyancy << std::endl;
  os << "center_of_gravity: " << model.center_of_gravity.transpose()
     << std::endl;
  os << "center_of_buoyancy: " << model.center_of_buoyancy.transpose()
     << std::endl;
  return os;
}

using SixDOFModel = Model<6>;

}  // namespace control
}  // namespace auv

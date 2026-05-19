#pragma once
#include <Eigen/Core>
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
  }

  // copy assignment
  Model& operator=(const Model& other) {
    if (this == &other) {
      return *this;
    }

    mass_inertia_matrix = other.mass_inertia_matrix;
    linear_damping_matrix = other.linear_damping_matrix;
    quadratic_damping_matrix = other.quadratic_damping_matrix;

    return *this;
  }

  // move constructor
  Model(Model&& other) noexcept {
    mass_inertia_matrix = std::move(other.mass_inertia_matrix);
    linear_damping_matrix = std::move(other.linear_damping_matrix);
    quadratic_damping_matrix = std::move(other.quadratic_damping_matrix);
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

  Matrix mass_inertia_matrix;
  // Matrix added_mass_matrix_; // TODO: implement this
  Matrix linear_damping_matrix;
  Matrix quadratic_damping_matrix;
  // Matrix coriolis_centrifugal_matrix_; // TODO: implement this
  // Vector gravity_vector_; // TODO: implement this

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
  return os;
}

using SixDOFModel = Model<6>;

}  // namespace control
}  // namespace auv

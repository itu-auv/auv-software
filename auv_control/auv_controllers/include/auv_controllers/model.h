#pragma once
#include <Eigen/Core>
#include <ostream>

namespace auv {
namespace control {

template <size_t N>
struct Model {
  using Matrix = Eigen::Matrix<double, N, N>;

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

  Matrix mass_inertia_matrix;
  // Matrix added_mass_matrix_; // TODO: implement this
  Matrix linear_damping_matrix;
  Matrix quadratic_damping_matrix;
  // Matrix coriolis_centrifugal_matrix_; // TODO: implement this
  // Vector gravity_vector_; // TODO: implement this
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
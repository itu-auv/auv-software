#pragma once
#include <Eigen/Dense>
#include <optional>

#include "fmt/format.h"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.hpp"
#include "tf2_ros/transform_listener.hpp"

namespace auv {
namespace control {

class ThrusterAllocator {
 public:
  static constexpr auto kDOF = 6;
  static constexpr auto kThrusterCount = 8;
  using AllocationMatrixColumn = Eigen::Matrix<double, kDOF, 1>;
  using AllocationMatrix = Eigen::Matrix<double, kDOF, kThrusterCount>;
  using TransposedAllocationMatrix =
      Eigen::Matrix<double, kThrusterCount, kDOF>;
  using Transform = std::pair<Eigen::Vector3d, Eigen::Quaterniond>;
  using WrenchVector = Eigen::Matrix<double, kDOF, 1>;
  using ThrusterEffortVector = Eigen::Matrix<double, kThrusterCount, 1>;
  using WrenchStampedVector =
      std::array<geometry_msgs::msg::WrenchStamped, kThrusterCount>;

  ThrusterAllocator(const rclcpp::Node::SharedPtr &nh)
      : nh_{nh}, tf_buffer_{nh->get_clock()}, tf_listener_{tf_buffer_} {
    frame_prefix_ =
        nh_->declare_parameter<std::string>("frame_prefix", "taluy");
    base_frame_ =
        nh_->declare_parameter<std::string>("base_frame", "base_link");
    thruster_frame_ = nh_->declare_parameter<std::string>("thruster_frame",
                                                          "thruster_{}_link");

    if (!frame_prefix_.empty()) {
      base_frame_ = frame_prefix_ + "/" + base_frame_;
      thruster_frame_ = frame_prefix_ + "/" + thruster_frame_;
    }

    const auto allocation_matrix = calculate_tam();

    if (!allocation_matrix) {
      RCLCPP_ERROR(nh_->get_logger(), "Failed to calculate TAM");
      return;
    }

    inverted_allocation_matrix_ = pseudo_inverse(allocation_matrix.value());
    RCLCPP_INFO(nh_->get_logger(), "ThrusterAllocator initialized");
    RCLCPP_INFO_STREAM(nh_->get_logger(),
                       "Inverted allocation matrix:\n"
                           << inverted_allocation_matrix_.value());
  }

  std::optional<ThrusterEffortVector> allocate(const WrenchVector &wrench) {
    if (!inverted_allocation_matrix_) {
      RCLCPP_ERROR(nh_->get_logger(), "Failed to allocate thruster forces");
      return std::nullopt;
    }
    return inverted_allocation_matrix_.value() * wrench;
  }

  void get_wrench_stamped_vector(const ThrusterEffortVector &effort,
                                 WrenchStampedVector &wrench_stamped_vector) {
    for (size_t i = 0; i < kThrusterCount; ++i) {
      auto &wrench_stamped = wrench_stamped_vector.at(i);
      wrench_stamped.header.stamp = nh_->now();
      wrench_stamped.header.frame_id = fmt::format(thruster_frame_, i);
      wrench_stamped.wrench.force.x = 0;
      wrench_stamped.wrench.force.y = 0;
      wrench_stamped.wrench.force.z = effort(i);
      wrench_stamped.wrench.torque.x = 0;
      wrench_stamped.wrench.torque.y = 0;
      wrench_stamped.wrench.torque.z = 0;
    }
  }

 private:
  std::optional<Transform> get_transform(const std::string &source_frame,
                                         const std::string &target_frame) {
    try {
      const auto transform = tf_buffer_.lookupTransform(
          target_frame, source_frame, tf2::TimePointZero,
          tf2::durationFromSec(1.0));

      const auto translation = Eigen::Vector3d{
          transform.transform.translation.x, transform.transform.translation.y,
          transform.transform.translation.z};
      const auto quaternion = Eigen::Quaterniond{
          transform.transform.rotation.w, transform.transform.rotation.x,
          transform.transform.rotation.y, transform.transform.rotation.z};

      return std::make_pair(translation, quaternion);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(nh_->get_logger(), "Failed to get transform: %s", ex.what());
      return std::nullopt;
    }
  }

  AllocationMatrixColumn calculate_tam_column(
      const Eigen::Vector3d &translation,
      const Eigen::Quaterniond &quaternion) {
    const auto local_force = Eigen::Vector3d::UnitZ();
    const Eigen::Matrix3d rotation_matrix = quaternion.toRotationMatrix();
    const auto global_force = rotation_matrix * local_force;
    const auto global_moment = translation.cross(global_force);

    AllocationMatrixColumn tam_column;
    tam_column.head<3>() = global_force;
    tam_column.tail<3>() = global_moment;
    return tam_column;
  }

  std::optional<AllocationMatrix> calculate_tam() {
    AllocationMatrix tam;

    for (size_t i = 0; i < kThrusterCount; ++i) {
      const auto transform =
          get_transform(fmt::format(thruster_frame_, i), base_frame_);

      if (!transform) {
        return std::nullopt;
      }

      tam.col(i) = calculate_tam_column(transform->first, transform->second);
    }
    return tam;
  }

  Eigen::MatrixXd pseudo_inverse(Eigen::MatrixXd matrix) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    float tolerance = 1.0e-6f * float(std::max(matrix.rows(), matrix.cols())) *
                      svd.singularValues().array().abs()(0);
    return svd.matrixV() *
           (svd.singularValues().array().abs() > tolerance)
               .select(svd.singularValues().array().inverse(), 0)
               .matrix()
               .asDiagonal() *
           svd.matrixU().adjoint();
  }

  rclcpp::Node::SharedPtr nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::optional<TransposedAllocationMatrix> inverted_allocation_matrix_;

  std::string frame_prefix_;
  std::string base_frame_;
  std::string thruster_frame_;
};

}  // namespace control
}  // namespace auv

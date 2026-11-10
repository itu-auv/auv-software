#pragma once

#include <tf2/LinearMath/Quaternion.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "rclcpp/rclcpp.hpp"

namespace auv_mapping {

/**
 * @brief ObjectPositionFilter uses an OpenCV Kalman Filter to track a 3D
 * position with a constant-velocity model, and applies quaternion slerp for
 * orientation.
 */
class ObjectPositionFilter {
 public:
  using Ptr = std::unique_ptr<ObjectPositionFilter>;

  /**
   * @brief Construct a new Object Filter object.
   * @param initial_transform The initial transform message.
   * @param dt The initial time step.
   */
  ObjectPositionFilter(
      const geometry_msgs::msg::TransformStamped &initial_transform,
      const double dt, rclcpp::Clock::SharedPtr clock);

  ~ObjectPositionFilter() = default;

  /// Predict the state forward by dt.
  void predict(const double dt);

  /**
   * @brief Update the filter with a new measurement.
   * @param measurement The incoming transform message.
   * @param dt Time step since last update.
   */
  void update(const geometry_msgs::msg::TransformStamped &measurement,
              const double dt);
  void updateFrameIndex(const std::string &new_frame_id);
  /// Get the filtered transform.
  geometry_msgs::msg::TransformStamped getFilteredTransform() const;

 private:
  /**
   * @brief Perform spherical linear interpolation between two quaternions.
   * @param q1 The starting quaternion.
   * @param q2 The target quaternion.
   * @param t Interpolation factor (0.0–1.0).
   * @return tf2::Quaternion The interpolated quaternion.
   */
  tf2::Quaternion slerp(const tf2::Quaternion &q1, const tf2::Quaternion &q2,
                        const double t) const;

  cv::KalmanFilter
      kf_;  // State: [x,y,z,vx,vy,vz] (6x1); Measurement: [x,y,z] (3x1)
  tf2::Quaternion filtered_orientation_;  // Filtered orientation
  std::string static_frame_;              // Parent (static) frame id
  std::string child_frame_;               // Child frame id for this object
  rclcpp::Clock::SharedPtr clock_;
};

}  // namespace auv_mapping

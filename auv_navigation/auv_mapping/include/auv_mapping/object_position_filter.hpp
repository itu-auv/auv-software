#pragma once

#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <string>

namespace auv_mapping {

/**
 * @brief ObjectPositionFilter uses an OpenCV Kalman Filter to track a 3D
 * position with a constant-velocity model, and applies quaternion slerp for
 * orientation.
 */
class ObjectPositionFilter {
 public:
  using Ptr = std::unique_ptr<ObjectPositionFilter>;

  struct AdaptiveNoiseConfig {
    double q_stddev = 0.01;
    double r_stddev = 0.1;
    double q_v_gain = 0.0;
    double q_omega_gain = 0.0;
    double r_v_gain = 0.0;
    double r_omega_gain = 0.0;
    double q_stddev_scale_min = 1.0;
    double q_stddev_scale_max = 1.0;
    double r_stddev_scale_min = 1.0;
    double r_stddev_scale_max = 1.0;
  };

  /**
   * @brief Construct a new Object Filter object.
   * @param initial_transform The initial transform message.
   * @param dt The initial time step.
   * @param noise_config Adaptive process/measurement noise parameters.
   */
  ObjectPositionFilter(const geometry_msgs::TransformStamped &initial_transform,
                       const double dt,
                       const AdaptiveNoiseConfig &noise_config);

  ~ObjectPositionFilter() = default;

  /// Predict the state forward by dt.
  void predict(const double dt);

  /**
   * @brief Update the filter with a new measurement.
   * @param measurement The incoming transform message.
   * @param dt Time step since last update.
   * @param v Vehicle translational speed magnitude.
   * @param omega Vehicle rotational speed magnitude.
   */
  void update(const geometry_msgs::TransformStamped &measurement,
              const double dt, const double v, const double omega);
  void updateFrameIndex(const std::string &new_frame_id);
  /// Get the filtered transform.
  geometry_msgs::TransformStamped getFilteredTransform() const;

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

  void updateNoiseCovariances(const double v, const double omega);
  double computeStddevScale(const double v, const double omega,
                            const double v_gain, const double omega_gain,
                            const double min_scale,
                            const double max_scale) const;

  cv::KalmanFilter
      kf_;  // State: [x,y,z,vx,vy,vz] (6x1); Measurement: [x,y,z] (3x1)
  AdaptiveNoiseConfig noise_config_;
  tf2::Quaternion filtered_orientation_;  // Filtered orientation
  std::string static_frame_;              // Parent (static) frame id
  std::string child_frame_;               // Child frame id for this object
};

}  // namespace auv_mapping

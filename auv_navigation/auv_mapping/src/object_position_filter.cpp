#include "auv_mapping/object_position_filter.hpp"

#include <algorithm>
#include <cmath>

namespace auv_mapping {

ObjectPositionFilter::ObjectPositionFilter(
    const geometry_msgs::TransformStamped &initial_transform, const double dt,
    const AdaptiveNoiseConfig &noise_config)
    : kf_(6, 3, 0),
      noise_config_(noise_config),
      static_frame_(initial_transform.header.frame_id),
      child_frame_(initial_transform.child_frame_id) {
  // Initialize state vector: [x, y, z, vx, vy, vz]
  kf_.statePre.at<float>(0) =
      static_cast<float>(initial_transform.transform.translation.x);
  kf_.statePre.at<float>(1) =
      static_cast<float>(initial_transform.transform.translation.y);
  kf_.statePre.at<float>(2) =
      static_cast<float>(initial_transform.transform.translation.z);
  kf_.statePre.at<float>(3) = 0.0f;
  kf_.statePre.at<float>(4) = 0.0f;
  kf_.statePre.at<float>(5) = 0.0f;
  kf_.statePost = kf_.statePre.clone();

  // Set up constant velocity model: F = [ I  dt*I; 0  I ]
  kf_.transitionMatrix =
      (cv::Mat_<float>(6, 6) << 1, 0, 0, dt, 0, 0, 0, 1, 0, 0, dt, 0, 0, 0, 1,
       0, 0, dt, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1);

  // Measurement matrix: H = [ I 0 ]
  kf_.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
  kf_.measurementMatrix.at<float>(0, 0) = 1.0f;
  kf_.measurementMatrix.at<float>(1, 1) = 1.0f;
  kf_.measurementMatrix.at<float>(2, 2) = 1.0f;

  updateNoiseCovariances(0.0, 0.0);

  // Initial error covariance
  kf_.errorCovPost = cv::Mat::eye(6, 6, CV_32F);

  // Initialize orientation from the initial transform.
  tf2::fromMsg(initial_transform.transform.rotation, filtered_orientation_);
  filtered_orientation_.normalize();
}

void ObjectPositionFilter::predict(const double dt) {
  // Update transition matrix with new dt.
  kf_.transitionMatrix.at<float>(0, 3) = static_cast<float>(dt);
  kf_.transitionMatrix.at<float>(1, 4) = static_cast<float>(dt);
  kf_.transitionMatrix.at<float>(2, 5) = static_cast<float>(dt);
  kf_.predict();
  // Orientation is assumed constant in prediction.
}

void ObjectPositionFilter::update(
    const geometry_msgs::TransformStamped &measurement, const double dt,
    const double v, const double omega) {
  updateNoiseCovariances(v, omega);
  predict(dt);

  // Create measurement vector [x, y, z]^T.
  cv::Mat meas(3, 1, CV_32F);
  meas.at<float>(0) = static_cast<float>(measurement.transform.translation.x);
  meas.at<float>(1) = static_cast<float>(measurement.transform.translation.y);
  meas.at<float>(2) = static_cast<float>(measurement.transform.translation.z);
  kf_.correct(meas);

  // Update orientation using slerp.
  tf2::Quaternion meas_q;
  tf2::fromMsg(measurement.transform.rotation, meas_q);
  meas_q.normalize();
  const double alpha = 0.2;  // Slerp weighting factor
  filtered_orientation_ = slerp(filtered_orientation_, meas_q, alpha);
  filtered_orientation_.normalize();
}

void ObjectPositionFilter::updateNoiseCovariances(const double v,
                                                  const double omega) {
  const double process_scale = computeStddevScale(
      v, omega, noise_config_.q_v_gain, noise_config_.q_omega_gain,
      noise_config_.q_stddev_scale_min, noise_config_.q_stddev_scale_max);
  const double measurement_scale = computeStddevScale(
      v, omega, noise_config_.r_v_gain, noise_config_.r_omega_gain,
      noise_config_.r_stddev_scale_min, noise_config_.r_stddev_scale_max);

  const double process_stddev = noise_config_.q_stddev * process_scale;
  const double measurement_stddev = noise_config_.r_stddev * measurement_scale;

  const float process_variance =
      static_cast<float>(process_stddev * process_stddev);
  const float measurement_variance =
      static_cast<float>(measurement_stddev * measurement_stddev);

  kf_.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * process_variance;
  kf_.measurementNoiseCov = cv::Mat::eye(3, 3, CV_32F) * measurement_variance;
}

double ObjectPositionFilter::computeStddevScale(const double v,
                                                const double omega,
                                                const double v_gain,
                                                const double omega_gain,
                                                const double min_scale,
                                                const double max_scale) const {
  const double scale = 1.0 + v_gain * v + omega_gain * omega;
  return std::clamp(scale, min_scale, max_scale);
}

void ObjectPositionFilter::updateFrameIndex(const std::string &new_frame_id) {
  child_frame_ = new_frame_id;
}
geometry_msgs::TransformStamped ObjectPositionFilter::getFilteredTransform()
    const {
  geometry_msgs::TransformStamped out;
  out.header.stamp = ros::Time::now();
  out.header.frame_id = static_frame_;
  out.child_frame_id = child_frame_;
  out.transform.translation.x = static_cast<double>(kf_.statePost.at<float>(0));
  out.transform.translation.y = static_cast<double>(kf_.statePost.at<float>(1));
  out.transform.translation.z = static_cast<double>(kf_.statePost.at<float>(2));
  out.transform.rotation = tf2::toMsg(filtered_orientation_);
  return out;
}

tf2::Quaternion ObjectPositionFilter::slerp(const tf2::Quaternion &q1,
                                            const tf2::Quaternion &q2,
                                            const double t) const {
  double dot = q1.dot(q2);
  tf2::Quaternion q2_mod = q2;
  if (dot < 0.0) {
    q2_mod = tf2::Quaternion(-q2.x(), -q2.y(), -q2.z(), -q2.w());
    dot = -dot;
  }
  constexpr double DOT_THRESHOLD = 0.9995;
  if (dot > DOT_THRESHOLD) {
    // Quaternions are nearly identical; use linear interpolation.
    tf2::Quaternion result = q1 + (q2_mod - q1) * t;
    result.normalize();
    return result;
  }
  const double theta_0 = acos(dot);
  const double theta = theta_0 * t;
  tf2::Quaternion q_perp = (q2_mod - q1 * dot);
  q_perp.normalize();
  return q1 * cos(theta) + q_perp * sin(theta);
}

}  // namespace auv_mapping

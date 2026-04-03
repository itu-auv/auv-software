
#include "auv_common_lib/ros/conversions.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "auv_common_lib/control/controller_types.h"
#include "geometry_msgs/Accel.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/Wrench.h"
#include "nav_msgs/Odometry.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

namespace auv {
namespace common {
namespace conversions {

template <>
Eigen::Vector3d convert(const geometry_msgs::Vector3& from) {
  return Eigen::Vector3d(from.x, from.y, from.z);
}

template <>
Eigen::Vector3d convert(const geometry_msgs::Point& from) {
  return Eigen::Vector3d(from.x, from.y, from.z);
}

template <>
geometry_msgs::Point convert(const Eigen::Vector3d& from) {
  geometry_msgs::Point to;
  to.x = from.x();
  to.y = from.y();
  to.z = from.z();
  return to;
}

template <>
Eigen::Vector3d convert(const geometry_msgs::Quaternion& from) {
  const auto quaternion = tf2::Quaternion(from.x, from.y, from.z, from.w);
  double roll, pitch, yaw;
  tf2::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);
  return Eigen::Vector3d(roll, pitch, yaw);
}

template <>
Eigen::Quaterniond convert(const geometry_msgs::Quaternion& from) {
  Eigen::Quaterniond to(from.w, from.x, from.y, from.z);
  if (to.norm() <= 1e-9) {
    return Eigen::Quaterniond::Identity();
  }

  to.normalize();
  return to;
}

template <>
geometry_msgs::Quaternion convert(const Eigen::Quaterniond& from) {
  const Eigen::Quaterniond normalized =
      from.norm() <= 1e-9 ? Eigen::Quaterniond::Identity() : from.normalized();

  geometry_msgs::Quaternion to;
  to.x = normalized.x();
  to.y = normalized.y();
  to.z = normalized.z();
  to.w = normalized.w();
  return to;
}

template <>
geometry_msgs::Vector3 convert(const Eigen::Vector3d& from) {
  geometry_msgs::Vector3 to;
  to.x = from.x();
  to.y = from.y();
  to.z = from.z();
  return to;
}

template <>
auv::control::SixDOFPose convert(const geometry_msgs::Pose& from) {
  auv::control::SixDOFPose to;
  to.position = convert<geometry_msgs::Point, Eigen::Vector3d>(from.position);
  to.orientation =
      convert<geometry_msgs::Quaternion, Eigen::Quaterniond>(from.orientation);
  to.normalize_orientation();
  return to;
}

template <>
auv::control::SixDOFTwist convert(const geometry_msgs::Twist& from) {
  auv::control::SixDOFTwist to;
  to.linear = convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.linear);
  to.angular = convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.angular);
  return to;
}

template <>
auv::control::SixDOFState convert(const nav_msgs::Odometry& from) {
  auv::control::SixDOFState to;
  to.pose =
      convert<geometry_msgs::Pose, auv::control::SixDOFPose>(from.pose.pose);
  to.twist = convert<geometry_msgs::Twist, auv::control::SixDOFTwist>(
      from.twist.twist);
  to.normalize_orientation();
  return to;
}

template <>
auv::control::SixDOFStateDerivative convert(const geometry_msgs::Accel& from) {
  auv::control::SixDOFStateDerivative to;
  to.linear_acceleration =
      convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.linear);
  to.angular_acceleration =
      convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.angular);
  return to;
}

template <>
Eigen::Matrix<double, 6, 1> convert(const geometry_msgs::Pose& from) {
  Eigen::Matrix<double, 6, 1> to;
  to.head<3>() = convert<geometry_msgs::Point, Eigen::Vector3d>(from.position);
  to.tail<3>() =
      convert<geometry_msgs::Quaternion, Eigen::Vector3d>(from.orientation);

  return to;
}

template <>
Eigen::Matrix<double, 6, 1> convert(
    const geometry_msgs::PoseWithCovariance& from) {
  return convert<geometry_msgs::Pose, Eigen::Matrix<double, 6, 1>>(from.pose);
}

template <>
Eigen::Matrix<double, 6, 1> convert(const geometry_msgs::Twist& from) {
  Eigen::Matrix<double, 6, 1> to;
  to.head<3>() = convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.linear);
  to.tail<3>() = convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.angular);

  return to;
}

template <>
Eigen::Matrix<double, 6, 1> convert(
    const geometry_msgs::TwistWithCovariance& from) {
  return convert<geometry_msgs::Twist, Eigen::Matrix<double, 6, 1>>(from.twist);
}

template <>
Eigen::Matrix<double, 6, 1> convert(const geometry_msgs::Wrench& from) {
  Eigen::Matrix<double, 6, 1> to;
  to.head<3>() = convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.force);
  to.tail<3>() = convert<geometry_msgs::Vector3, Eigen::Vector3d>(from.torque);
  return to;
}

template <>
Eigen::Matrix<double, 12, 1> convert(const nav_msgs::Odometry& from) {
  Eigen::Matrix<double, 12, 1> to;
  to.head<6>() =
      convert<geometry_msgs::Pose, Eigen::Matrix<double, 6, 1>>(from.pose.pose);
  to.tail<6>() = convert<geometry_msgs::Twist, Eigen::Matrix<double, 6, 1>>(
      from.twist.twist);

  return to;
}

template <>
geometry_msgs::Wrench convert(const Eigen::Matrix<double, 6, 1>& from) {
  geometry_msgs::Wrench to;
  to.force = convert<Eigen::Vector3d, geometry_msgs::Vector3>(from.head<3>());
  to.torque = convert<Eigen::Vector3d, geometry_msgs::Vector3>(from.tail<3>());

  return to;
}

}  // namespace conversions
}  // namespace common
}  // namespace auv

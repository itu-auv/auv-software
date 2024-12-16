
#include "auv_common_lib/ros/conversions.h"

#include <Eigen/Dense>

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
geometry_msgs::Vector3 convert(const Eigen::Vector3d& from) {
  geometry_msgs::Vector3 to;
  to.x = from.x();
  to.y = from.y();
  to.z = from.z();
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

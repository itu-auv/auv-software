#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/Geometry.h"
#include "canbus/f16_converter.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class IMUReportModule : public ModuleBase {
  static constexpr auto kOrientationReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::OrientationReport);
  static constexpr auto kAngularVelocityReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::AngularVelocityReport);
  static constexpr auto kLinearAccelerationReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::LinearAccelerationReport);

 public:
  IMUReportModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    imu_report_publisher =
        ModuleBase::node_handle().advertise<sensor_msgs::Imu>(
            "mainboard/imu/data", 10);
    ROS_INFO_STREAM("Initialized IMUReportModule for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override {
    static auto imu_ros_msg = []() {
      sensor_msgs::Imu msg;
      msg.orientation_covariance.fill(0.1);
      msg.angular_velocity_covariance.fill(0.1);
      msg.linear_acceleration_covariance.fill(0.1);
      msg.header.frame_id = "imu_frame"; // TODO: Set correct frame_id
      return msg;
    }();

    const auto geometry_msg = deserialize_message<auv_canbus_msgs::Geometry>(
        data.data(), data.size());  // ??

    switch (id) {
      case kOrientationReportIdentifier: {
        imu_ros_msg.header.stamp = ros::Time::now();
        imu_ros_msg.header.seq++;

        imu_ros_msg.orientation.x =
            auv::math::from_float16(geometry_msg.orientation.x);
        imu_ros_msg.orientation.y =
            auv::math::from_float16(geometry_msg.orientation.y);
        imu_ros_msg.orientation.z =
            auv::math::from_float16(geometry_msg.orientation.z);
        imu_ros_msg.orientation.w =
            auv::math::from_float16(geometry_msg.orientation.w);

        imu_report_publisher.publish(imu_ros_msg);
        break;
      }
      case kAngularVelocityReportIdentifier: {
        imu_ros_msg.header.stamp = ros::Time::now();
        imu_ros_msg.header.seq++;

        imu_ros_msg.angular_velocity.x =
            auv::math::from_float16(geometry_msg.angular_velocity.x);
        imu_ros_msg.angular_velocity.y =
            auv::math::from_float16(geometry_msg.angular_velocity.y);
        imu_ros_msg.angular_velocity.z =
            auv::math::from_float16(geometry_msg.angular_velocity.z);
        imu_report_publisher.publish(imu_ros_msg);

        break;
      }
      case kLinearAccelerationReportIdentifier: {
        imu_ros_msg.header.stamp = ros::Time::now();
        imu_ros_msg.header.seq++;
        imu_ros_msg.linear_acceleration.x =
            auv::math::from_float16(geometry_msg.linear_acceleration.x);
        imu_ros_msg.linear_acceleration.y =
            auv::math::from_float16(geometry_msg.linear_acceleration.y);
        imu_ros_msg.linear_acceleration.z =
            auv::math::from_float16(geometry_msg.linear_acceleration.z);
        imu_report_publisher.publish(imu_ros_msg);
        break;
      }
      default:
    }
  };

 private:
  ros::Publisher imu_report_publisher;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

#pragma once
#include <algorithm>
#include <optional>
#include <vector>

#include "auv_control/thruster_allocation.h"
#include "auv_msgs/MotorCommand.h"
#include "auv_msgs/Power.h"
#include "ros/ros.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace auv {
namespace control {

class ThrusterManagerROS {
 public:
  static constexpr auto kDOF = ThrusterAllocator::kDOF;
  static constexpr auto kThrusterCount = ThrusterAllocator::kThrusterCount;
  using WrenchVector = ThrusterAllocator::WrenchVector;
  using ThrusterEffortVector = ThrusterAllocator::ThrusterEffortVector;

  ThrusterManagerROS(const ros::NodeHandle &nh)
      : nh_{nh},
        allocator_{ros::NodeHandle{"~"}},
        tf_buffer_{},
        tf_listener_{tf_buffer_} {
    ROS_INFO("ThrusterManagerROS initialized");

    ros::NodeHandle nh_private("~");
    nh_private.getParam("coeffs_ccw", coeffs_ccw_);
    nh_private.getParam("coeffs_cw", coeffs_cw_);
    nh_private.getParam("mapping", mapping_);
    nh_private.getParam("max_thrust", max_wrench_);
    nh_private.getParam("min_thrust", min_wrench_);
    nh_private.param<std::string>("body_frame", body_frame_, "taluy/base_link");
    nh_private.param<double>("transform_timeout", transform_timeout_, 1.0);

    for (size_t i = 0; i < kThrusterCount; ++i) {
      thruster_wrench_pubs_[i] = nh_.advertise<geometry_msgs::WrenchStamped>(
          "thrusters/thruster_" + std::to_string(i) + "/wrench", 1);
    }

    const auto transport_hints = ros::TransportHints().tcpNoDelay(true);

    wrench_sub_ =
        nh_.subscribe("wrench", 1, &ThrusterManagerROS::wrench_callback, this,
                      transport_hints);

    power_sub_ =
        nh_.subscribe("power", 1, &ThrusterManagerROS::power_callback, this);

    drive_pub_ = nh_.advertise<auv_msgs::MotorCommand>("board/drive_pulse", 1);
  }

  void spin() {
    ros::Rate rate(10);
    while (ros::ok()) {
      if (!latest_wrench_ || !latest_power_) {
        ros::spinOnce();
        rate.sleep();
        continue;
      }

      const auto wrench_msg = latest_wrench_.value();

      // Transform wrench if frame_id is provided
      geometry_msgs::WrenchStamped transformed_wrench = wrench_msg;
      if (!wrench_msg.header.frame_id.empty() &&
          wrench_msg.header.frame_id != body_frame_) {
        try {
          // Check if transform is available
          tf_buffer_.canTransform(body_frame_, wrench_msg.header.frame_id,
                                  ros::Time(0),
                                  ros::Duration(transform_timeout_));

          // Transform the wrench
          tf2::doTransform(
              wrench_msg, transformed_wrench,
              tf_buffer_.lookupTransform(
                  body_frame_, wrench_msg.header.frame_id, ros::Time(0)));
        } catch (const tf2::TransformException &ex) {
          ROS_WARN_STREAM("Could not transform wrench from "
                          << wrench_msg.header.frame_id << " to " << body_frame_
                          << ": " << ex.what());
          ros::spinOnce();
          rate.sleep();
          continue;
        }
      }

      auto thruster_efforts =
          allocator_.allocate(to_vector(transformed_wrench.wrench));
      allocator_.get_wrench_stamped_vector(thruster_efforts.value(),
                                           thruster_wrench_msgs_);

      ThrusterEffortVector efforts = ThrusterEffortVector::Zero();
      if (thruster_efforts && !is_timeouted()) {
        efforts = thruster_efforts.value();
      }

      for (size_t i = 0; i < kThrusterCount; ++i) {
        thruster_wrench_pubs_[i].publish(thruster_wrench_msgs_.at(i));
      }

      auv_msgs::MotorCommand motor_command_msg;
      double voltage = latest_power_ ? latest_power_->voltage : 16.0;
      for (size_t i = 0; i < kThrusterCount; ++i) {
        const auto pwm = wrench_to_drive(efforts(mapping_[i]), voltage);
        motor_command_msg.channels[i] = std::clamp(
            pwm, static_cast<uint16_t>(1100U), static_cast<uint16_t>(1900U));
      }

      drive_pub_.publish(motor_command_msg);

      ros::spinOnce();
      rate.sleep();
    }
  }

 private:
  WrenchVector to_vector(const geometry_msgs::Wrench &wrench) {
    WrenchVector vector;
    vector << wrench.force.x, wrench.force.y, wrench.force.z, wrench.torque.x,
        wrench.torque.y, wrench.torque.z;
    return vector;
  }

  bool is_timeouted() const {
    return (ros::Time::now() - latest_wrench_time_).toSec() > 1.0;
  }

  void wrench_callback(const geometry_msgs::WrenchStamped &msg) {
    latest_wrench_ = msg;
    latest_wrench_time_ = ros::Time::now();
  }

  void power_callback(const auv_msgs::Power &msg) { latest_power_ = msg; }

  uint16_t wrench_to_drive(double wrench, double voltage) {
    if (std::abs(wrench) < min_wrench_) {
      return 1500;
    }

    if (std::abs(wrench) > max_wrench_) {
      wrench = std::copysign(max_wrench_, wrench);
    }

    wrench /= 9.81;

    const auto &coeffs = (wrench > 0) ? coeffs_cw_ : coeffs_ccw_;
    const auto drive_value = calculate_drive_value(coeffs, wrench, voltage);
    return std::clamp(drive_value, static_cast<uint16_t>(1100U),
                      static_cast<uint16_t>(1900U));
  }

  uint16_t calculate_drive_value(const std::vector<double> &coeffs,
                                 double wrench, double voltage) const {
    double a = coeffs[0], b = coeffs[1], c = coeffs[2], d = coeffs[3],
           e = coeffs[4], f = coeffs[5];
    double drive_value = a * wrench * wrench + b * wrench * voltage +
                         c * voltage * voltage + d * wrench + e * voltage + f;
    return static_cast<uint16_t>(drive_value);
  }

  ros::NodeHandle nh_;
  ThrusterAllocator allocator_;
  std::optional<geometry_msgs::WrenchStamped> latest_wrench_;
  std::optional<auv_msgs::Power> latest_power_;
  ros::Time latest_wrench_time_{ros::Time(0)};
  std::array<ros::Publisher, kThrusterCount> thruster_wrench_pubs_;
  std::array<geometry_msgs::WrenchStamped, kThrusterCount>
      thruster_wrench_msgs_;
  ros::Subscriber wrench_sub_;
  ros::Subscriber power_sub_;
  ros::Publisher drive_pub_;

  // TF related members
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string body_frame_;
  double transform_timeout_;

  std::vector<double> coeffs_ccw_;
  std::vector<double> coeffs_cw_;
  std::vector<int> mapping_;
  double max_wrench_{0.0};
  double min_wrench_{0.0};
};

}  // namespace control
}  // namespace auv

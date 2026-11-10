#pragma once
#include <algorithm>
#include <optional>
#include <vector>

#include "auv_control/thruster_allocation.h"
#include "rclcpp/create_timer.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/battery_state.hpp"
#include "std_msgs/msg/u_int16_multi_array.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
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

  ThrusterManagerROS(const rclcpp::Node::SharedPtr &nh)
      : nh_{nh},
        allocator_{nh},
        tf_buffer_{nh->get_clock()},
        tf_listener_{tf_buffer_} {
    RCLCPP_INFO(nh_->get_logger(), "ThrusterManagerROS initialized");

    coeffs_ccw_ = nh_->declare_parameter<std::vector<double>>(
        "coeffs_ccw", std::vector<double>{});
    coeffs_cw_ = nh_->declare_parameter<std::vector<double>>(
        "coeffs_cw", std::vector<double>{});
    mapping_ = nh_->declare_parameter<std::vector<int64_t>>(
        "mapping", std::vector<int64_t>{});
    max_wrench_ = nh_->declare_parameter<double>("max_thrust", 0.0);
    min_wrench_ = nh_->declare_parameter<double>("min_thrust", 0.0);
    voltage_threshold_ =
        nh_->declare_parameter<double>("minimum_voltage_threshold", 13.5);
    low_voltage_duration_ =
        nh_->declare_parameter<double>("low_voltage_duration", 3.0);

    // Load per-thruster direction: 1 = normal, -1 = reversed (CCW)
    directions_ = nh_->declare_parameter<std::vector<int64_t>>(
        "directions", std::vector<int64_t>(kThrusterCount, 1));
    body_frame_ =
        nh_->declare_parameter<std::string>("body_frame", "taluy/base_link");
    transform_timeout_ =
        nh_->declare_parameter<double>("transform_timeout", 1.0);

    for (size_t i = 0; i < kThrusterCount; ++i) {
      thruster_wrench_pubs_[i] =
          nh_->create_publisher<geometry_msgs::msg::WrenchStamped>(
              "thrusters/thruster_" + std::to_string(i) + "/wrench", 1);
    }
    wrench_sub_ = nh_->create_subscription<geometry_msgs::msg::WrenchStamped>(
        "wrench", 1,
        std::bind(&ThrusterManagerROS::wrench_callback, this,
                  std::placeholders::_1));
    power_sub_ = nh_->create_subscription<sensor_msgs::msg::BatteryState>(
        "power", 1,
        std::bind(&ThrusterManagerROS::power_callback, this,
                  std::placeholders::_1));
    drive_pub_ = nh_->create_publisher<std_msgs::msg::UInt16MultiArray>(
        "board/drive_pulse", 1);
  }

  void spin() {
    auto timer = rclcpp::create_timer(
        nh_, nh_->get_clock(), rclcpp::Duration::from_seconds(0.1), [this]() {
          if (!latest_power_) {
            return;
          }

          const auto voltage = latest_power_->voltage;
          if (voltage < voltage_threshold_) {
            if (!low_voltage_start_) {
              low_voltage_start_ = nh_->now();
            }

            if ((nh_->now() - low_voltage_start_.value()).seconds() >=
                low_voltage_duration_) {
              RCLCPP_ERROR_STREAM(
                  nh_->get_logger(),
                  "Battery voltage stayed below "
                      << voltage_threshold_ << " V for "
                      << low_voltage_duration_
                      << " seconds. Stopping thruster manager.");
              rclcpp::shutdown();
              return;
            }
          } else {
            low_voltage_start_.reset();
          }

          if (!latest_wrench_) {
            return;
          }

          const auto wrench_msg = latest_wrench_.value();

          // Transform wrench if frame_id is provided
          geometry_msgs::msg::WrenchStamped transformed_wrench = wrench_msg;
          if (!wrench_msg.header.frame_id.empty() &&
              wrench_msg.header.frame_id != body_frame_) {
            try {
              // Check if transform is available
              tf_buffer_.canTransform(body_frame_, wrench_msg.header.frame_id,
                                      tf2::TimePointZero,
                                      tf2::durationFromSec(transform_timeout_));

              // Transform the wrench
              tf2::doTransform(wrench_msg, transformed_wrench,
                               tf_buffer_.lookupTransform(
                                   body_frame_, wrench_msg.header.frame_id,
                                   tf2::TimePointZero));
            } catch (const tf2::TransformException &ex) {
              RCLCPP_WARN_STREAM(nh_->get_logger(),
                                 "Could not transform wrench from "
                                     << wrench_msg.header.frame_id << " to "
                                     << body_frame_ << ": " << ex.what());
              return;
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
            thruster_wrench_pubs_[i]->publish(thruster_wrench_msgs_.at(i));
          }

          std_msgs::msg::UInt16MultiArray motor_command_msg;
          motor_command_msg.data.resize(kThrusterCount);
          for (size_t i = 0; i < kThrusterCount; ++i) {
            const int dir = (i < directions_.size()) ? directions_[i] : 1;
            const auto pwm =
                wrench_to_drive(efforts(mapping_[i]) * dir, voltage);
            motor_command_msg.data[i] =
                std::clamp(pwm, static_cast<uint16_t>(1100U),
                           static_cast<uint16_t>(1900U));
          }

          drive_pub_->publish(motor_command_msg);
        });
    rclcpp::spin(nh_);
  }

 private:
  WrenchVector to_vector(const geometry_msgs::msg::Wrench &wrench) {
    WrenchVector vector;
    vector << wrench.force.x, wrench.force.y, wrench.force.z, wrench.torque.x,
        wrench.torque.y, wrench.torque.z;
    return vector;
  }

  bool is_timeouted() const {
    return (nh_->now() - latest_wrench_time_).seconds() > 1.0;
  }

  void wrench_callback(
      const geometry_msgs::msg::WrenchStamped::ConstSharedPtr msg) {
    latest_wrench_ = *msg;
    latest_wrench_time_ = nh_->now();
  }

  void power_callback(
      const sensor_msgs::msg::BatteryState::ConstSharedPtr msg) {
    latest_power_ = *msg;
  }

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

  rclcpp::Node::SharedPtr nh_;
  ThrusterAllocator allocator_;
  std::optional<geometry_msgs::msg::WrenchStamped> latest_wrench_;
  std::optional<sensor_msgs::msg::BatteryState> latest_power_;
  rclcpp::Time latest_wrench_time_{0, 0, RCL_ROS_TIME};
  std::array<rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr,
             kThrusterCount>
      thruster_wrench_pubs_;
  std::array<geometry_msgs::msg::WrenchStamped, kThrusterCount>
      thruster_wrench_msgs_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr
      wrench_sub_;
  rclcpp::Subscription<sensor_msgs::msg::BatteryState>::SharedPtr power_sub_;
  rclcpp::Publisher<std_msgs::msg::UInt16MultiArray>::SharedPtr drive_pub_;

  // TF related members
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string body_frame_;
  double transform_timeout_;
  double voltage_threshold_{14.0};
  double low_voltage_duration_{5.0};
  std::optional<rclcpp::Time> low_voltage_start_;

  std::vector<double> coeffs_ccw_;
  std::vector<double> coeffs_cw_;
  std::vector<int64_t> mapping_;
  std::vector<int64_t> directions_;
  double max_wrench_{0.0};
  double min_wrench_{0.0};
};

}  // namespace control
}  // namespace auv

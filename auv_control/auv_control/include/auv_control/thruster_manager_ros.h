#pragma once
#include "auv_control/thruster_allocation.h"
#include "ros/ros.h"
#include "auv_msgs/MotorCommand.h"
#include <vector>

namespace auv {
namespace control {

class ThrusterManagerROS {
 public:
  static constexpr auto kDOF = ThrusterAllocator::kDOF;
  static constexpr auto kThrusterCount = ThrusterAllocator::kThrusterCount;
  using WrenchVector = ThrusterAllocator::WrenchVector;
  using ThrusterEffortVector = ThrusterAllocator::ThrusterEffortVector;

  ThrusterManagerROS(const ros::NodeHandle &nh)
      : nh_{nh}, allocator_{ros::NodeHandle{"~"}} {
    ROS_INFO("ThrusterManagerROS initialized");

    ros::NodeHandle nh_private("~");
    nh_private.getParam("coeffs_ccw", coeffs_ccw_);
    nh_private.getParam("coeffs_cw", coeffs_cw_);

    for (size_t i = 0; i < kThrusterCount; ++i) {
      thruster_wrench_pubs_[i] = nh_.advertise<geometry_msgs::WrenchStamped>(
          "thrusters/thruster_" + std::to_string(i) + "/wrench", 1);
    }
    wrench_sub_ =
        nh_.subscribe("wrench", 1, &ThrusterManagerROS::wrench_callback, this);

    drive_pub_ = nh_.advertise<auv_msgs::MotorCommand>("/drive", 1);
  }

  void spin() {
    ros::Rate rate(10);
    while (ros::ok()) {
      if (!latest_wrench_) {
        ros::spinOnce();
        rate.sleep();
        continue;
      }
      const auto wrench_msg = latest_wrench_.value();
      auto thruster_efforts = allocator_.allocate(to_vector(wrench_msg));
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
      for (size_t i = 0; i < kThrusterCount; ++i) {
        motor_command_msg.channels[i] = wrench_to_drive(efforts(i));; 
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

  void wrench_callback(const geometry_msgs::Wrench &msg) {
    latest_wrench_ = msg;
    latest_wrench_time_ = ros::Time::now();
  }

  uint16_t wrench_to_drive(double wrench, double voltage = 16.0) {
    if (std::abs(wrench) < 0.05) {
        return 1500;
    }

    double a, b, c, d, e, f;
    if (wrench > 0) {
        a = coeffs_cw_[0];
        b = coeffs_cw_[1];
        c = coeffs_cw_[2];
        d = coeffs_cw_[3];
        e = coeffs_cw_[4];
        f = coeffs_cw_[5];
    } else {
        a = coeffs_ccw_[0];
        b = coeffs_ccw_[1];
        c = coeffs_ccw_[2];
        d = coeffs_ccw_[3];
        e = coeffs_ccw_[4];
        f = coeffs_ccw_[5];
    }

    double drive_value = a * wrench * wrench + b * wrench * voltage + c * voltage * voltage + d * wrench + e * voltage + f;
    return static_cast<uint16_t>(drive_value);
}

  ros::NodeHandle nh_;
  ThrusterAllocator allocator_;
  std::optional<geometry_msgs::Wrench> latest_wrench_;
  ros::Time latest_wrench_time_{ros::Time(0)};
  std::array<ros::Publisher, kThrusterCount> thruster_wrench_pubs_;
  std::array<geometry_msgs::WrenchStamped, kThrusterCount> thruster_wrench_msgs_;
  ros::Subscriber wrench_sub_;
  ros::Publisher drive_pub_;  

  std::vector<double> coeffs_ccw_;
  std::vector<double> coeffs_cw_;
};

}  // namespace control
}  // namespace auv

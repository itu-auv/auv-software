#pragma once
#include "auv_control/thruster_allocation.h"
#include "ros/ros.h"

namespace auv {
namespace control {
//

class ThrusterManagerROS {
 public:
  static constexpr auto kDOF = ThrusterAllocator::kDOF;
  static constexpr auto kThrusterCount = ThrusterAllocator::kThrusterCount;
  using WrenchVector = ThrusterAllocator::WrenchVector;
  using ThrusterEffortVector = ThrusterAllocator::ThrusterEffortVector;

  ThrusterManagerROS(const ros::NodeHandle &nh)
      : nh_{nh}, allocator_{ros::NodeHandle{"~"}} {
    ROS_INFO("ThrusterManagerROS initialized");

    for (size_t i = 0; i < kThrusterCount; ++i) {
      thruster_wrench_pubs_[i] = nh_.advertise<geometry_msgs::WrenchStamped>(
          "thrusters/thruster_" + std::to_string(i) + "/wrench", 1);
    }
    wrench_sub_ =
        nh_.subscribe("wrench", 1, &ThrusterManagerROS::wrench_callback, this);
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

  ros::NodeHandle nh_;
  ThrusterAllocator allocator_;
  std::optional<geometry_msgs::Wrench> latest_wrench_;
  ros::Time latest_wrench_time_{ros::Time(0)};
  std::array<ros::Publisher, kThrusterCount> thruster_wrench_pubs_;
  std::array<geometry_msgs::WrenchStamped, kThrusterCount>
      thruster_wrench_msgs_;
  ros::Subscriber wrench_sub_;
};

}  // namespace control
}  // namespace auv
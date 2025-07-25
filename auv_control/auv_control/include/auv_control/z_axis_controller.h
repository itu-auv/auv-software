#pragma once

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Wrench.h>
#include <geometry_msgs/WrenchStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>

#include "auv_common_lib/ros/subscriber_with_timeout.h"

namespace auv {
namespace control {

class ZAxisController {
 public:
  ZAxisController(const ros::NodeHandle& nh);
  void spin();

 private:
  using ControlEnableSub =
      auv::common::ros::SubscriberWithTimeout<std_msgs::Bool>;

  bool is_control_enabled();
  bool is_timeouted() const;
  void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void wrenchCallback(const geometry_msgs::Wrench::ConstPtr& msg);
  void cmdPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);

  // A simple PID controller implementation
  class PID {
   public:
    PID(double p, double i, double d)
        : kp_(p), ki_(i), kd_(d), integral_(0), prev_error_(0) {}
    double control(double current, double desired, double dt);
    void setGains(double p, double i, double d) {
      kp_ = p;
      ki_ = i;
      kd_ = d;
    }

   private:
    double kp_, ki_, kd_;
    double integral_;
    double prev_error_;
  };

  ros::NodeHandle nh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber wrench_sub_;
  ros::Subscriber cmd_pose_sub_;
  ros::Publisher wrench_pub_;
  ControlEnableSub control_enable_sub_;
  ros::Rate rate_;
  ros::Time latest_command_time_{ros::Time(0)};

  double current_z_ = 0.0;
  double desired_z_ = 0.0;
  geometry_msgs::Wrench latest_wrench_;

  PID z_pid_;
  std::string body_frame_;
};

}  // namespace control
}  // namespace auv

#pragma once

#include <angles/angles.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Wrench.h>
#include <geometry_msgs/WrenchStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include "auv_common_lib/ros/subscriber_with_timeout.h"
#include "auv_control/WrenchControllerConfig.h"

namespace auv {
namespace control {

class WrenchController {
 public:
  WrenchController(const ros::NodeHandle& nh);
  void spin();

 private:
  using ControlEnableSub =
      auv::common::ros::SubscriberWithTimeout<std_msgs::Bool>;

  bool is_control_enabled();
  bool is_timeouted() const;
  void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void wrenchCallback(const geometry_msgs::Wrench::ConstPtr& msg);
  void cmdPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void reconfigureCallback(const auv_control::WrenchControllerConfig& config,
                           uint32_t level);

  // A simple PID controller implementation
  class PID {
   public:
    PID(double p, double i, double d)
        : kp_(p), ki_(i), kd_(d), integral_(0), prev_error_(0) {}
    void reset() {
      integral_ = 0.0;
      prev_error_ = 0.0;
    }
    double control(double current, double desired, double dt);
    double controlFromError(double error, double dt);
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
  double current_roll_ = 0.0;
  double desired_roll_ = 0.0;
  double current_pitch_ = 0.0;
  double desired_pitch_ = 0.0;
  double current_yaw_ = 0.0;
  double desired_yaw_ = 0.0;
  geometry_msgs::Wrench latest_wrench_;

  PID z_pid_;
  PID roll_pid_;
  PID pitch_pid_;
  PID yaw_pid_;
  std::string body_frame_;

  dynamic_reconfigure::Server<auv_control::WrenchControllerConfig>
      reconfigure_server_;
};

}  // namespace control
}  // namespace auv

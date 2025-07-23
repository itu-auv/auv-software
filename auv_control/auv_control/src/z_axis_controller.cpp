#include "auv_control/z_axis_controller.h"

namespace auv {
namespace control {

ZAxisController::ZAxisController(const ros::NodeHandle& nh)
    : nh_(nh), rate_(20.0), z_pid_(0.0, 0.0, 0.0) {
  ros::NodeHandle nh_private("~");

  double p, i, d;
  nh_private.param("p_gain", p, 0.0);
  nh_private.param("i_gain", i, 0.0);
  nh_private.param("d_gain", d, 0.0);
  z_pid_.setGains(p, i, d);

  nh_private.param<std::string>("body_frame", body_frame_,
                                "taluy_mini/base_link");

  odom_sub_ =
      nh_.subscribe("odometry", 1, &ZAxisController::odometryCallback, this);
  wrench_sub_ =
      nh_.subscribe("wrench_cmd", 1, &ZAxisController::wrenchCallback, this);
  cmd_pose_sub_ =
      nh_.subscribe("cmd_pose", 1, &ZAxisController::cmdPoseCallback, this);
  wrench_pub_ = nh_.advertise<geometry_msgs::WrenchStamped>("wrench", 1);
}

void ZAxisController::odometryCallback(
    const nav_msgs::Odometry::ConstPtr& msg) {
  current_z_ = msg->pose.pose.position.z;
}

void ZAxisController::wrenchCallback(
    const geometry_msgs::Wrench::ConstPtr& msg) {
  latest_wrench_ = *msg;
}

void ZAxisController::cmdPoseCallback(
    const geometry_msgs::PoseStamped::ConstPtr& msg) {
  desired_z_ = msg->pose.position.z;
}

double ZAxisController::PID::control(double current, double desired,
                                     double dt) {
  double error = desired - current;
  integral_ += error * dt;
  double derivative = (error - prev_error_) / dt;
  prev_error_ = error;
  return kp_ * error + ki_ * integral_ + kd_ * derivative;
}

void ZAxisController::spin() {
  while (ros::ok()) {
    ros::spinOnce();

    double z_force = z_pid_.control(current_z_, desired_z_,
                                    rate_.expectedCycleTime().toSec());

    geometry_msgs::WrenchStamped wrench_msg;
    wrench_msg.header.stamp = ros::Time::now();
    wrench_msg.header.frame_id = body_frame_;
    wrench_msg.wrench = latest_wrench_;
    wrench_msg.wrench.force.z = z_force;

    wrench_pub_.publish(wrench_msg);

    rate_.sleep();
  }
}

}  // namespace control
}  // namespace auv

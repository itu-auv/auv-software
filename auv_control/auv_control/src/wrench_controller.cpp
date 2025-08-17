#include "auv_control/wrench_controller.h"

namespace auv {
namespace control {

WrenchController::WrenchController(const ros::NodeHandle& nh)
    : nh_(nh),
      rate_(20.0),
      z_pid_(0.0, 0.0, 0.0),
      control_enable_sub_{nh_},
      reconfigure_server_{ros::NodeHandle("~")} {
  ros::NodeHandle nh_private("~");
  reconfigure_server_.setCallback(
      boost::bind(&WrenchController::reconfigureCallback, this, _1, _2));

  nh_private.param<std::string>("body_frame", body_frame_,
                                "taluy_mini/base_link");

  odom_sub_ =
      nh_.subscribe("odometry", 1, &WrenchController::odometryCallback, this);
  cmd_vel_sub_ =
      nh_.subscribe("cmd_vel", 1, &WrenchController::twistCallback, this);
  cmd_pose_sub_ =
      nh_.subscribe("cmd_pose", 1, &WrenchController::cmdPoseCallback, this);
  wrench_pub_ = nh_.advertise<geometry_msgs::WrenchStamped>("wrench", 1);
  control_enable_sub_.subscribe(
      "enable", 1, nullptr,
      []() { ROS_WARN_STREAM("control enable message timeouted"); },
      ros::Duration{1.0});
  control_enable_sub_.set_default_message(std_msgs::Bool{});
}

void WrenchController::odometryCallback(
    const nav_msgs::Odometry::ConstPtr& msg) {
  current_z_ = msg->pose.pose.position.z;
}

void WrenchController::twistCallback(
    const geometry_msgs::Twist::ConstPtr& msg) {
  latest_cmd_vel_ = *msg;
  latest_command_time_ = ros::Time::now();
}

void WrenchController::cmdPoseCallback(
    const geometry_msgs::PoseStamped::ConstPtr& msg) {
  desired_z_ = msg->pose.position.z;
  latest_command_time_ = ros::Time::now();
}

void WrenchController::reconfigureCallback(
    const auv_control::WrenchControllerConfig& config, uint32_t level) {
  z_pid_.setGains(config.z_p_gain, config.z_i_gain, config.z_d_gain);
  zx_multiplier_ = config.zx_multiplier;
  zy_multiplier_ = config.zy_multiplier;
  yyaw_multiplier_ = config.yyaw_multiplier;
  yx_multiplier_ = config.yx_multiplier;
  linear_xy_scalar_ = config.linear_xy_scalar;
  roll_scalar_ = config.roll_scalar;
  pitch_scalar_ = config.pitch_scalar;
  yaw_scalar_ = config.yaw_scalar;
}

bool WrenchController::is_control_enabled() {
  bool enabled = control_enable_sub_.get_message().data;
  if (!enabled) {
    z_pid_.reset();
  }
  return enabled;
}

bool WrenchController::is_timeouted() const {
  return (ros::Time::now() - latest_command_time_).toSec() > 1.0;
}

double WrenchController::PID::control(double current, double desired,
                                      double dt) {
  double error = desired - current;
  integral_ += error * dt;
  double derivative = (error - prev_error_) / dt;
  prev_error_ = error;
  return kp_ * error + ki_ * integral_ + kd_ * derivative;
}

void WrenchController::spin() {
  while (ros::ok()) {
    ros::spinOnce();

    if (is_control_enabled() && !is_timeouted()) {
      const auto dt = rate_.expectedCycleTime().toSec();

      double z_force = z_pid_.control(current_z_, desired_z_, dt);

      geometry_msgs::WrenchStamped wrench_msg;
      wrench_msg.header.stamp = ros::Time::now();
      wrench_msg.header.frame_id = body_frame_;
      double y_force = latest_cmd_vel_.linear.y * linear_xy_scalar_ +
                       zy_multiplier_ * z_force;
      wrench_msg.wrench.force.x = latest_cmd_vel_.linear.x * linear_xy_scalar_ +
                                  zx_multiplier_ * z_force +
                                  yx_multiplier_ * y_force;
      wrench_msg.wrench.force.y = y_force;
      wrench_msg.wrench.force.z = z_force;
      wrench_msg.wrench.torque.x = latest_cmd_vel_.angular.x * roll_scalar_;
      wrench_msg.wrench.torque.y = latest_cmd_vel_.angular.y * pitch_scalar_;
      wrench_msg.wrench.torque.z =
          latest_cmd_vel_.angular.z * yaw_scalar_ + yyaw_multiplier_ * y_force;
      wrench_pub_.publish(wrench_msg);
    }

    rate_.sleep();
  }
}

}  // namespace control
}  // namespace auv

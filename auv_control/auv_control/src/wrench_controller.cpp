#include "auv_control/wrench_controller.h"

namespace auv {
namespace control {

WrenchController::WrenchController(const ros::NodeHandle& nh)
    : nh_(nh),
      rate_(20.0),
      z_pid_(0.0, 0.0, 0.0),
      roll_pid_(0.0, 0.0, 0.0),
      pitch_pid_(0.0, 0.0, 0.0),
      yaw_pos_pid_(0.0, 0.0, 0.0),
      yaw_vel_pid_(0.0, 0.0, 0.0),
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
  tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  tf2::Matrix3x3 m(q);
  m.getRPY(current_roll_, current_pitch_, current_yaw_);
  current_yaw_vel_ = msg->twist.twist.angular.z;
}

void WrenchController::twistCallback(
    const geometry_msgs::Twist::ConstPtr& msg) {
  latest_cmd_vel_ = *msg;
  latest_command_time_ = ros::Time::now();
}

void WrenchController::cmdPoseCallback(
    const geometry_msgs::PoseStamped::ConstPtr& msg) {
  desired_z_ = msg->pose.position.z;
  tf2::Quaternion q(msg->pose.orientation.x, msg->pose.orientation.y,
                    msg->pose.orientation.z, msg->pose.orientation.w);
  tf2::Matrix3x3 m(q);
  m.getRPY(desired_roll_, desired_pitch_, desired_yaw_);
  latest_command_time_ = ros::Time::now();
}

void WrenchController::reconfigureCallback(
    const auv_control::WrenchControllerConfig& config, uint32_t level) {
  z_pid_.setGains(config.z_p_gain, config.z_i_gain, config.z_d_gain);
  roll_pid_.setGains(config.roll_p_gain, config.roll_i_gain,
                     config.roll_d_gain);
  pitch_pid_.setGains(config.pitch_p_gain, config.pitch_i_gain,
                      config.pitch_d_gain);
  yaw_pos_pid_.setGains(config.yaw_pos_p_gain, config.yaw_pos_i_gain,
                        config.yaw_pos_d_gain);
  yaw_vel_pid_.setGains(config.yaw_vel_p_gain, config.yaw_vel_i_gain,
                        config.yaw_vel_d_gain);
  zx_multiplier_ = config.zx_multiplier;
  zy_multiplier_ = config.zy_multiplier;
  linear_xy_scalar_ = config.linear_xy_scalar;
  linear_z_scalar_ = config.linear_z_scalar;
}

bool WrenchController::is_control_enabled() {
  bool enabled = control_enable_sub_.get_message().data;
  if (!enabled) {
    z_pid_.reset();
    roll_pid_.reset();
    pitch_pid_.reset();
    yaw_pos_pid_.reset();
    yaw_vel_pid_.reset();
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

double WrenchController::PID::controlFromError(double error, double dt) {
  integral_ += error * dt;
  double derivative = (error - prev_error_) / dt;
  prev_error_ = error;
  return kp_ * error + ki_ * integral_ + kd_ * derivative;
}

void WrenchController::spin() {
  while (ros::ok()) {
    ros::spinOnce();

    if (is_control_enabled() && !is_timeouted()) {
      double z_force = z_pid_.control(current_z_, desired_z_,
                                      rate_.expectedCycleTime().toSec());
      double roll_torque = roll_pid_.control(current_roll_, desired_roll_,
                                             rate_.expectedCycleTime().toSec());
      double pitch_torque = pitch_pid_.control(
          current_pitch_, desired_pitch_, rate_.expectedCycleTime().toSec());

      double yaw_pos_error =
          angles::shortest_angular_distance(current_yaw_, desired_yaw_);
      double desired_yaw_vel = yaw_pos_pid_.controlFromError(
          yaw_pos_error, rate_.expectedCycleTime().toSec());
      double yaw_torque = yaw_vel_pid_.control(
          current_yaw_vel_, desired_yaw_vel + latest_cmd_vel_.angular.z,
          rate_.expectedCycleTime().toSec());

      geometry_msgs::WrenchStamped wrench_msg;
      wrench_msg.header.stamp = ros::Time::now();
      wrench_msg.header.frame_id = body_frame_;

      wrench_msg.wrench.force.z = z_force;
      wrench_msg.wrench.force.x = latest_cmd_vel_.linear.x * linear_xy_scalar_ +
                                  wrench_msg.wrench.force.z * zx_multiplier_;
      wrench_msg.wrench.force.y = latest_cmd_vel_.linear.y * linear_xy_scalar_ +
                                  wrench_msg.wrench.force.z * zy_multiplier_;

      wrench_msg.wrench.torque.x = roll_torque;
      wrench_msg.wrench.torque.y = pitch_torque;
      wrench_msg.wrench.torque.z = yaw_torque;

      wrench_pub_.publish(wrench_msg);
    }

    rate_.sleep();
  }
}

}  // namespace control
}  // namespace auv

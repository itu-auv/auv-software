#pragma once

#include <auv_control/ControllerConfig.h>  // Include your dynamic reconfigure header
#include <dynamic_reconfigure/server.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <type_traits>
#include <vector>

#include "auv_common_lib/ros/conversions.h"
#include "auv_common_lib/ros/rosparam.h"
#include "auv_common_lib/ros/subscriber_with_timeout.h"
#include "auv_controllers/controller_base.h"
#include "auv_controllers/multidof_pid_controller.h"
#include "geometry_msgs/AccelWithCovarianceStamped.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Wrench.h"
#include "nav_msgs/Odometry.h"
#include "pluginlib/class_loader.h"
#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float64.h"
#include "tf2_ros/buffer.h"

namespace auv {
namespace control {

class ControllerROS {
 public:
  using ControllerBase = SixDOFControllerBase;
  using Model = SixDOFModel;
  using ModelParser = auv::common::rosparam::parser<Model>;
  using MatrixRosparamParser =
      auv::common::rosparam::parser<ControllerBase::Matrix>;
  using VectorRosparamParser =
      auv::common::rosparam::parser<Eigen::Matrix<double, 12, 1>>;
  using Vector6RosparamParser =
      auv::common::rosparam::parser<Eigen::Matrix<double, 6, 1>>;
  using ControllerLoader = pluginlib::ClassLoader<SixDOFControllerBase>;
  using ControllerBasePtr =
      boost::shared_ptr<auv::control::SixDOFControllerBase>;
  using ControlEnableSub =
      auv::common::ros::SubscriberWithTimeout<std_msgs::Bool>;

  ControllerROS(const ros::NodeHandle& nh)
      : nh_{nh},
        rate_{1.0},
        control_enable_sub_{nh},
        tf_buffer_{},
        tf_listener_{tf_buffer_} {
    ros::NodeHandle nh_private("~");

    // default_frame as a ros parameter (default is odom)
    depth_control_reference_frame_ =
        nh_private.param<std::string>("depth_control_reference_frame", "odom");

    auto model = ModelParser::parse("model", nh_private);
    load_parameters();

    ROS_INFO_STREAM("Model: \n" << model);

    const auto rate = nh_private.param("rate", 10.0);
    rate_ = ros::Rate{rate};

    nh_private.param<std::string>("body_frame", body_frame_, "taluy/base_link");
    nh_private.param<double>("transform_timeout", transform_timeout_, 1.0);
    nh_private.param<double>("odometry_timeout", odometry_timeout_, 1.0);
    nh_private.param<double>("dvl_invalid_velocity_zero_timeout",
                             dvl_invalid_velocity_zero_timeout_, 1.0);

    ROS_INFO_STREAM("kp: \n" << kp_.transpose());
    ROS_INFO_STREAM("ki: \n" << ki_.transpose());
    ROS_INFO_STREAM("kd: \n" << kd_.transpose());
    ROS_INFO_STREAM("dvl_invalid_position_kp_xy: \n"
                    << dvl_invalid_position_kp_xy_.transpose());
    ROS_INFO_STREAM("dvl_invalid_position_ki_xy: \n"
                    << dvl_invalid_position_ki_xy_.transpose());
    ROS_INFO_STREAM("dvl_invalid_position_kd_xy: \n"
                    << dvl_invalid_position_kd_xy_.transpose());
    ROS_INFO_STREAM("dvl_invalid_velocity_kp_xy: \n"
                    << dvl_invalid_velocity_kp_xy_.transpose());
    ROS_INFO_STREAM("dvl_invalid_velocity_ki_xy: \n"
                    << dvl_invalid_velocity_ki_xy_.transpose());
    ROS_INFO_STREAM("dvl_invalid_velocity_kd_xy: \n"
                    << dvl_invalid_velocity_kd_xy_.transpose());
    ROS_INFO_STREAM("integral_clamp_limits: \n"
                    << integral_clamp_limits_.transpose());
    ROS_INFO_STREAM("gravity_compensation_z: " << gravity_compensation_z_);
    load_controller("auv::control::SixDOFPIDController");

    auto controller =
        dynamic_cast<auv::control::SixDOFPIDController*>(controller_.get());

    controller->set_model(model);
    controller->set_kp(kp_);
    controller->set_ki(ki_);
    controller->set_kd(kd_);
    controller->set_integral_clamp_limits(integral_clamp_limits_);
    controller->set_gravity_compensation_z(gravity_compensation_z_);
    controller->set_max_velocity_limits(max_velocity_);
    controller->set_max_acceleration_limits(max_acceleration_);
    controller->set_max_acceleration_rate_limits(max_acceleration_rate_);

    // Set up dynamic reconfigure server with initial values
    auv_control::ControllerConfig initial_config;
    set_initial_config(initial_config);

    dynamic_reconfigure::Server<auv_control::ControllerConfig>::CallbackType f;
    f = boost::bind(&ControllerROS::reconfigure_callback, this, _1, _2);
    dr_srv_.updateConfig(initial_config);  // Apply the initial configuration
    dr_srv_.setCallback(f);

    const auto transport_hints = ros::TransportHints().tcpNoDelay(true);

    odometry_sub_ =
        nh_.subscribe("odometry", 1, &ControllerROS::odometry_callback, this,
                      transport_hints);
    cmd_vel_sub_ = nh_.subscribe("cmd_vel", 1, &ControllerROS::cmd_vel_callback,
                                 this, transport_hints);
    cmd_pose_sub_ =
        nh_.subscribe("cmd_pose", 1, &ControllerROS::cmd_pose_callback, this,
                      transport_hints);
    accel_sub_ =
        nh_.subscribe("acceleration", 1, &ControllerROS::accel_callback, this,
                      transport_hints);
    dvl_is_valid_sub_ =
        nh_.subscribe("dvl/is_valid", 1, &ControllerROS::dvl_is_valid_callback,
                      this, transport_hints);

    control_enable_sub_.subscribe(
        "enable", 1, nullptr,
        []() { ROS_WARN_STREAM("control enable message timeouted"); },
        ros::Duration{1.0});
    control_enable_sub_.set_default_message(std_msgs::Bool{});

    wrench_pub_ = nh_.advertise<geometry_msgs::WrenchStamped>("wrench", 1);
    desired_velocity_pub_ =
        nh_.advertise<geometry_msgs::Twist>("desired_velocity", 1);
  }

  bool load_controller(const std::string& controller_name) {
    try {
      controller_ = controller_loader_.createInstance(controller_name);
      ROS_INFO_STREAM("Controller loaded: " << controller_name);

      return true;
    } catch (pluginlib::PluginlibException& ex) {
      ROS_ERROR("The plugin failed to load for some reason. Error: %s",
                ex.what());
      return false;
    }
  }

  void spin() {
    const auto dt = 1.0 / rate_.expectedCycleTime().toSec();

    while (ros::ok()) {
      ros::spinOnce();
      rate_.sleep();

      if (!controller_) {
        continue;
      }

      if ((ros::Time::now() - latest_cmd_vel_time_).toSec() > 1.0) {
        desired_state_.tail(6) = ControllerBase::Vector::Zero();
      }

      auto* pid_controller =
          dynamic_cast<auv::control::SixDOFPIDController*>(controller_.get());
      if (pid_controller) {
        const auto use_zero_velocity_state =
            should_use_zero_velocity_state_for_velocity_error();
        pid_controller->set_use_zero_velocity_state_for_velocity_error(
            use_zero_velocity_state);
        apply_dvl_invalid_gain_schedule(pid_controller,
                                        use_zero_velocity_state);

        if (use_zero_velocity_state) {
          ROS_WARN_THROTTLE(
              1.0,
              "DVL has been invalid for at least %.2f seconds; using zero "
              "velocity state for velocity error",
              dvl_invalid_velocity_zero_timeout_);
        }
      }

      const auto control_output =
          controller_->control(state_, desired_state_, d_state_, dt);

      if (pid_controller) {
        desired_velocity_pub_.publish(
            auv::common::conversions::convert<ControllerBase::Vector,
                                              geometry_msgs::Twist>(
                pid_controller->get_desired_velocity()));
      }

      geometry_msgs::WrenchStamped wrench_msg;
      if (is_control_enabled() && !is_timeouted() && has_fresh_odometry()) {
        wrench_msg.header.stamp = ros::Time::now();
        wrench_msg.header.frame_id = body_frame_;
        wrench_msg.wrench =
            auv::common::conversions::convert<ControllerBase::WrenchVector,
                                              geometry_msgs::Wrench>(
                control_output);
        wrench_pub_.publish(wrench_msg);
      } else if (is_control_enabled() && !has_fresh_odometry()) {
        ROS_WARN_THROTTLE(3.0,
                          "control enable requested but odometry is missing or "
                          "stale, not publishing wrench");
      }
    }
  }

 private:
  bool is_control_enabled() { return control_enable_sub_.get_message().data; }
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string body_frame_;
  double transform_timeout_;
  double odometry_timeout_;
  bool is_timeouted() const {
    const auto latest_time =
        std::max(latest_cmd_vel_time_, latest_cmd_pose_time_);
    return (ros::Time::now() - latest_time).toSec() > 1.0;
  }

  bool has_fresh_odometry() const {
    if (latest_odometry_time_.isZero()) {
      return false;
    }

    return (ros::Time::now() - latest_odometry_time_).toSec() <=
           odometry_timeout_;
  }

  void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    state_ =
        auv::common::conversions::convert<nav_msgs::Odometry,
                                          ControllerBase::StateVector>(*msg);
    d_state_.head(6) = state_.tail(6);
    latest_odometry_time_ =
        msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
  }

  void cmd_vel_callback(const geometry_msgs::Twist::ConstPtr& msg) {
    if ((ros::Time::now() - latest_cmd_pose_time_).toSec() > 1.0) {
      desired_state_.head(6) = state_.head(6);
    }

    desired_state_.tail(6) =
        auv::common::conversions::convert<geometry_msgs::Twist,
                                          ControllerBase::Vector>(*msg);
    latest_cmd_vel_time_ = ros::Time::now();
  }

  const std::optional<std::string> get_source_frame(
      const std::string& source_frame) {
    // no transform will be needed with an empty frame (assume odom frame)
    if (source_frame.empty()) {
      return std::nullopt;
    }

    // no transform will be needed between two identical frames
    if (source_frame == depth_control_reference_frame_) {
      return std::nullopt;
    }

    // Transform is required: remove leading slash if present
    if (source_frame[0] == '/') {
      return source_frame.substr(1);
    }
    return source_frame;
  }

  void cmd_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    const auto source_frame = get_source_frame(msg->header.frame_id);
    auto transformed_pose = msg->pose;
    const auto lookup_time = msg->header.stamp;

    static tf2_ros::Buffer tf_buffer;
    static tf2_ros::TransformListener tf_listener(tf_buffer);
    geometry_msgs::TransformStamped transform_stamped;

    if (source_frame.has_value()) {
      ROS_DEBUG("Source frame: %s, Desired frame: %s",
                source_frame.value().c_str(),
                depth_control_reference_frame_.c_str());
      try {
        transform_stamped = tf_buffer.lookupTransform(
            depth_control_reference_frame_, source_frame.value(), lookup_time,
            ros::Duration(transform_timeout_));

        tf2::doTransform(msg->pose, transformed_pose, transform_stamped);
      } catch (tf2::TransformException& ex) {
        ROS_DEBUG("Failed to transform pose");
        return;
      }
    }
    ROS_DEBUG_STREAM("Final transformed command pose: "
                     << transformed_pose.position.x << ", "
                     << transformed_pose.position.y << ", "
                     << transformed_pose.position.z);

    desired_state_.head(6) = auv::common::conversions::convert<
        geometry_msgs::Pose, ControllerBase::Vector>(transformed_pose);

    latest_cmd_pose_time_ =
        lookup_time.isZero() ? ros::Time::now() : lookup_time;
  }

  void accel_callback(
      const geometry_msgs::AccelWithCovarianceStamped::ConstPtr& msg) {
    d_state_(6) = msg->accel.accel.linear.x;
    d_state_(7) = msg->accel.accel.linear.y;
    d_state_(8) = msg->accel.accel.linear.z;
    d_state_.tail(3) = Eigen::Vector3d::Zero();
  }

  void dvl_is_valid_callback(const std_msgs::Bool::ConstPtr& msg) {
    has_dvl_is_valid_message_ = true;

    if (msg->data) {
      dvl_is_valid_ = true;
      dvl_invalid_since_ = ros::Time(0);
      return;
    }

    if (dvl_is_valid_ || dvl_invalid_since_.isZero()) {
      dvl_invalid_since_ = ros::Time::now();
    }
    dvl_is_valid_ = false;
  }

  bool should_use_zero_velocity_state_for_velocity_error() const {
    if (!has_dvl_is_valid_message_ || dvl_is_valid_ ||
        dvl_invalid_since_.isZero()) {
      return false;
    }

    return (ros::Time::now() - dvl_invalid_since_).toSec() >=
           dvl_invalid_velocity_zero_timeout_;
  }

  void apply_dvl_invalid_gain_schedule(
      auv::control::SixDOFPIDController* controller,
      const bool use_dvl_invalid_gains) const {
    auto active_kp = kp_;
    auto active_ki = ki_;
    auto active_kd = kd_;

    if (use_dvl_invalid_gains) {
      active_kp(0) = dvl_invalid_position_kp_xy_(0);
      active_kp(1) = dvl_invalid_position_kp_xy_(1);
      active_ki(0) = dvl_invalid_position_ki_xy_(0);
      active_ki(1) = dvl_invalid_position_ki_xy_(1);
      active_kd(0) = dvl_invalid_position_kd_xy_(0);
      active_kd(1) = dvl_invalid_position_kd_xy_(1);

      active_kp(6) = dvl_invalid_velocity_kp_xy_(0);
      active_kp(7) = dvl_invalid_velocity_kp_xy_(1);
      active_ki(6) = dvl_invalid_velocity_ki_xy_(0);
      active_ki(7) = dvl_invalid_velocity_ki_xy_(1);
      active_kd(6) = dvl_invalid_velocity_kd_xy_(0);
      active_kd(7) = dvl_invalid_velocity_kd_xy_(1);
    }

    controller->set_kp(active_kp);
    controller->set_ki(active_ki);
    controller->set_kd(active_kd);
  }

  void reconfigure_callback(auv_control::ControllerConfig& config,
                            uint32_t level) {
    auto controller =
        dynamic_cast<auv::control::SixDOFPIDController*>(controller_.get());

    kp_ << config.kp_0, config.kp_1, config.kp_2, config.kp_3, config.kp_4,
        config.kp_5, config.kp_6, config.kp_7, config.kp_8, config.kp_9,
        config.kp_10, config.kp_11;
    ki_ << config.ki_0, config.ki_1, config.ki_2, config.ki_3, config.ki_4,
        config.ki_5, config.ki_6, config.ki_7, config.ki_8, config.ki_9,
        config.ki_10, config.ki_11;
    kd_ << config.kd_0, config.kd_1, config.kd_2, config.kd_3, config.kd_4,
        config.kd_5, config.kd_6, config.kd_7, config.kd_8, config.kd_9,
        config.kd_10, config.kd_11;
    integral_clamp_limits_ << config.integral_clamp_0, config.integral_clamp_1,
        config.integral_clamp_2, config.integral_clamp_3,
        config.integral_clamp_4, config.integral_clamp_5,
        config.integral_clamp_6, config.integral_clamp_7,
        config.integral_clamp_8, config.integral_clamp_9,
        config.integral_clamp_10, config.integral_clamp_11;
    dvl_invalid_position_kp_xy_ << config.dvl_invalid_position_kp_x,
        config.dvl_invalid_position_kp_y;
    dvl_invalid_position_ki_xy_ << config.dvl_invalid_position_ki_x,
        config.dvl_invalid_position_ki_y;
    dvl_invalid_position_kd_xy_ << config.dvl_invalid_position_kd_x,
        config.dvl_invalid_position_kd_y;
    dvl_invalid_velocity_kp_xy_ << config.dvl_invalid_velocity_kp_x,
        config.dvl_invalid_velocity_kp_y;
    dvl_invalid_velocity_ki_xy_ << config.dvl_invalid_velocity_ki_x,
        config.dvl_invalid_velocity_ki_y;
    dvl_invalid_velocity_kd_xy_ << config.dvl_invalid_velocity_kd_x,
        config.dvl_invalid_velocity_kd_y;
    apply_dvl_invalid_gain_schedule(
        controller, should_use_zero_velocity_state_for_velocity_error());
    controller->set_integral_clamp_limits(integral_clamp_limits_);
    controller->set_gravity_compensation_z(config.gravity_compensation_z);
    gravity_compensation_z_ = config.gravity_compensation_z;

    max_velocity_ << config.max_velocity_0, config.max_velocity_1,
        config.max_velocity_2, config.max_velocity_3, config.max_velocity_4,
        config.max_velocity_5;
    controller->set_max_velocity_limits(max_velocity_);

    max_acceleration_ << config.max_acceleration_0, config.max_acceleration_1,
        config.max_acceleration_2, config.max_acceleration_3,
        config.max_acceleration_4, config.max_acceleration_5;
    controller->set_max_acceleration_limits(max_acceleration_);

    max_acceleration_rate_ << config.max_acceleration_rate_0,
        config.max_acceleration_rate_1, config.max_acceleration_rate_2,
        config.max_acceleration_rate_3, config.max_acceleration_rate_4,
        config.max_acceleration_rate_5;
    controller->set_max_acceleration_rate_limits(max_acceleration_rate_);

    save_parameters();
  }

  void load_parameters() {
    kp_ = VectorRosparamParser::parse("kp", ros::NodeHandle("~"));
    ki_ = VectorRosparamParser::parse("ki", ros::NodeHandle("~"));
    kd_ = VectorRosparamParser::parse("kd", ros::NodeHandle("~"));

    // Load integral clamp limits with default values of 0 (0 means no clamping)
    ros::NodeHandle nh_private("~");
    dvl_invalid_position_kp_xy_ << kp_(0), kp_(1);
    dvl_invalid_position_ki_xy_ << ki_(0), ki_(1);
    dvl_invalid_position_kd_xy_ << kd_(0), kd_(1);
    dvl_invalid_velocity_kp_xy_ << kp_(6), kp_(7);
    dvl_invalid_velocity_ki_xy_ << ki_(6), ki_(7);
    dvl_invalid_velocity_kd_xy_ << kd_(6), kd_(7);
    load_optional_xy_gains("dvl_invalid_position_kp_xy",
                           dvl_invalid_position_kp_xy_, nh_private);
    load_optional_xy_gains("dvl_invalid_position_ki_xy",
                           dvl_invalid_position_ki_xy_, nh_private);
    load_optional_xy_gains("dvl_invalid_position_kd_xy",
                           dvl_invalid_position_kd_xy_, nh_private);
    load_optional_xy_gains("dvl_invalid_velocity_kp_xy",
                           dvl_invalid_velocity_kp_xy_, nh_private);
    load_optional_xy_gains("dvl_invalid_velocity_ki_xy",
                           dvl_invalid_velocity_ki_xy_, nh_private);
    load_optional_xy_gains("dvl_invalid_velocity_kd_xy",
                           dvl_invalid_velocity_kd_xy_, nh_private);

    if (nh_private.hasParam("integral_clamp_limits")) {
      integral_clamp_limits_ = VectorRosparamParser::parse(
          "integral_clamp_limits", ros::NodeHandle("~"));
      ROS_INFO("Loaded integral_clamp_limits parameter");
    } else {
      // If parameter doesn't exist, no clamping
      integral_clamp_limits_ = Eigen::Matrix<double, 12, 1>::Zero();
      ROS_INFO(
          "No integral_clamp_limits parameter found, integral clamping "
          "disabled");
    }

    // Load gravity compensation parameter
    gravity_compensation_z_ = nh_private.param("gravity_compensation_z", 0.0);

    // Load max velocity limits
    if (nh_private.hasParam("max_velocity")) {
      max_velocity_ = Vector6RosparamParser::parse("max_velocity", nh_private);
      ROS_INFO_STREAM("Loaded max_velocity: " << max_velocity_.transpose());
    } else {
      max_velocity_ = Eigen::Matrix<double, 6, 1>::Constant(1e6);
      ROS_WARN_STREAM("No max_velocity parameter found, limits disabled");
    }

    // Load max acceleration command limits
    if (nh_private.hasParam("max_acceleration")) {
      max_acceleration_ =
          Vector6RosparamParser::parse("max_acceleration", nh_private);
      ROS_INFO_STREAM(
          "Loaded max_acceleration: " << max_acceleration_.transpose());
    } else {
      max_acceleration_ = Eigen::Matrix<double, 6, 1>::Zero();
      ROS_INFO_STREAM(
          "No max_acceleration parameter found, acceleration limits disabled");
    }

    // Load max acceleration command rate limits
    if (nh_private.hasParam("max_acceleration_rate")) {
      max_acceleration_rate_ =
          Vector6RosparamParser::parse("max_acceleration_rate", nh_private);
      ROS_INFO_STREAM("Loaded max_acceleration_rate: "
                      << max_acceleration_rate_.transpose());
    } else {
      max_acceleration_rate_ = Eigen::Matrix<double, 6, 1>::Zero();
      ROS_INFO_STREAM(
          "No max_acceleration_rate parameter found, acceleration rate limits "
          "disabled");
    }
  }

  void load_optional_xy_gains(const std::string& param_name,
                              Eigen::Vector2d& gains,
                              const ros::NodeHandle& nh_private) {
    std::vector<double> values;
    if (!nh_private.getParam(param_name, values)) {
      return;
    }

    if (values.size() != 2) {
      ROS_WARN_STREAM("Ignoring " << param_name
                                  << ": expected exactly 2 values for x/y");
      return;
    }

    gains << values[0], values[1];
  }

  void set_initial_config(auv_control::ControllerConfig& config) {
    config.kp_0 = kp_(0);
    config.kp_1 = kp_(1);
    config.kp_2 = kp_(2);
    config.kp_3 = kp_(3);
    config.kp_4 = kp_(4);
    config.kp_5 = kp_(5);
    config.kp_6 = kp_(6);
    config.kp_7 = kp_(7);
    config.kp_8 = kp_(8);
    config.kp_9 = kp_(9);
    config.kp_10 = kp_(10);
    config.kp_11 = kp_(11);

    config.ki_0 = ki_(0);
    config.ki_1 = ki_(1);
    config.ki_2 = ki_(2);
    config.ki_3 = ki_(3);
    config.ki_4 = ki_(4);
    config.ki_5 = ki_(5);
    config.ki_6 = ki_(6);
    config.ki_7 = ki_(7);
    config.ki_8 = ki_(8);
    config.ki_9 = ki_(9);
    config.ki_10 = ki_(10);
    config.ki_11 = ki_(11);

    config.kd_0 = kd_(0);
    config.kd_1 = kd_(1);
    config.kd_2 = kd_(2);
    config.kd_3 = kd_(3);
    config.kd_4 = kd_(4);
    config.kd_5 = kd_(5);
    config.kd_6 = kd_(6);
    config.kd_7 = kd_(7);
    config.kd_8 = kd_(8);
    config.kd_9 = kd_(9);
    config.kd_10 = kd_(10);
    config.kd_11 = kd_(11);

    config.dvl_invalid_position_kp_x = dvl_invalid_position_kp_xy_(0);
    config.dvl_invalid_position_kp_y = dvl_invalid_position_kp_xy_(1);
    config.dvl_invalid_position_ki_x = dvl_invalid_position_ki_xy_(0);
    config.dvl_invalid_position_ki_y = dvl_invalid_position_ki_xy_(1);
    config.dvl_invalid_position_kd_x = dvl_invalid_position_kd_xy_(0);
    config.dvl_invalid_position_kd_y = dvl_invalid_position_kd_xy_(1);
    config.dvl_invalid_velocity_kp_x = dvl_invalid_velocity_kp_xy_(0);
    config.dvl_invalid_velocity_kp_y = dvl_invalid_velocity_kp_xy_(1);
    config.dvl_invalid_velocity_ki_x = dvl_invalid_velocity_ki_xy_(0);
    config.dvl_invalid_velocity_ki_y = dvl_invalid_velocity_ki_xy_(1);
    config.dvl_invalid_velocity_kd_x = dvl_invalid_velocity_kd_xy_(0);
    config.dvl_invalid_velocity_kd_y = dvl_invalid_velocity_kd_xy_(1);

    config.integral_clamp_0 = integral_clamp_limits_(0);
    config.integral_clamp_1 = integral_clamp_limits_(1);
    config.integral_clamp_2 = integral_clamp_limits_(2);
    config.integral_clamp_3 = integral_clamp_limits_(3);
    config.integral_clamp_4 = integral_clamp_limits_(4);
    config.integral_clamp_5 = integral_clamp_limits_(5);
    config.integral_clamp_6 = integral_clamp_limits_(6);
    config.integral_clamp_7 = integral_clamp_limits_(7);
    config.integral_clamp_8 = integral_clamp_limits_(8);
    config.integral_clamp_9 = integral_clamp_limits_(9);
    config.integral_clamp_10 = integral_clamp_limits_(10);
    config.integral_clamp_11 = integral_clamp_limits_(11);

    config.gravity_compensation_z = gravity_compensation_z_;

    config.max_velocity_0 = max_velocity_(0);
    config.max_velocity_1 = max_velocity_(1);
    config.max_velocity_2 = max_velocity_(2);
    config.max_velocity_3 = max_velocity_(3);
    config.max_velocity_4 = max_velocity_(4);
    config.max_velocity_5 = max_velocity_(5);

    config.max_acceleration_0 = max_acceleration_(0);
    config.max_acceleration_1 = max_acceleration_(1);
    config.max_acceleration_2 = max_acceleration_(2);
    config.max_acceleration_3 = max_acceleration_(3);
    config.max_acceleration_4 = max_acceleration_(4);
    config.max_acceleration_5 = max_acceleration_(5);

    config.max_acceleration_rate_0 = max_acceleration_rate_(0);
    config.max_acceleration_rate_1 = max_acceleration_rate_(1);
    config.max_acceleration_rate_2 = max_acceleration_rate_(2);
    config.max_acceleration_rate_3 = max_acceleration_rate_(3);
    config.max_acceleration_rate_4 = max_acceleration_rate_(4);
    config.max_acceleration_rate_5 = max_acceleration_rate_(5);
  }

  void save_parameters() {
    ros::NodeHandle nh_private("~");
    nh_private.param("config_file", config_file_, std::string{});
    if (config_file_.empty()) {
      ROS_ERROR("Config file not specified");
      return;
    }

    std::ifstream in_file(config_file_);
    if (!in_file.is_open()) {
      ROS_ERROR_STREAM("Failed to open config file: " << config_file_);
      return;
    }

    std::stringstream buffer;
    buffer << in_file.rdbuf();
    std::string content = buffer.str();
    in_file.close();

    auto replace_param = [](std::string& content, const std::string& param,
                            const Eigen::Matrix<double, 12, 1>& values) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(1);
      ss << param << ": [" << values(0);
      for (int i = 1; i < 12; ++i) ss << ", " << values(i);
      ss << "]";

      std::string::size_type start_pos = content.find(param + ": [");
      if (start_pos == std::string::npos) {
        // If parameter not found, add it to the end
        content += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = content.find("]", start_pos);
        content.replace(start_pos, end_pos - start_pos + 1, ss.str());
      }
    };

    replace_param(content, "kp", kp_);
    replace_param(content, "ki", ki_);
    replace_param(content, "kd", kd_);
    replace_param(content, "integral_clamp_limits", integral_clamp_limits_);

    auto replace_vector2_param = [](std::string& content,
                                    const std::string& param,
                                    const Eigen::Vector2d& values) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(3);
      ss << param << ": [" << values(0) << ", " << values(1) << "]";

      std::string::size_type start_pos = content.find(param + ": [");
      if (start_pos == std::string::npos) {
        content += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = content.find("]", start_pos);
        content.replace(start_pos, end_pos - start_pos + 1, ss.str());
      }
    };

    replace_vector2_param(content, "dvl_invalid_position_kp_xy",
                          dvl_invalid_position_kp_xy_);
    replace_vector2_param(content, "dvl_invalid_position_ki_xy",
                          dvl_invalid_position_ki_xy_);
    replace_vector2_param(content, "dvl_invalid_position_kd_xy",
                          dvl_invalid_position_kd_xy_);
    replace_vector2_param(content, "dvl_invalid_velocity_kp_xy",
                          dvl_invalid_velocity_kp_xy_);
    replace_vector2_param(content, "dvl_invalid_velocity_ki_xy",
                          dvl_invalid_velocity_ki_xy_);
    replace_vector2_param(content, "dvl_invalid_velocity_kd_xy",
                          dvl_invalid_velocity_kd_xy_);

    auto replace_vector6_param = [](std::string& content,
                                    const std::string& param,
                                    const Eigen::Matrix<double, 6, 1>& values) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(3);
      ss << param << ": [" << values(0);
      for (int i = 1; i < 6; ++i) ss << ", " << values(i);
      ss << "]";

      std::string::size_type start_pos = content.find(param + ": [");
      if (start_pos == std::string::npos) {
        content += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = content.find("]", start_pos);
        content.replace(start_pos, end_pos - start_pos + 1, ss.str());
      }
    };

    replace_vector6_param(content, "max_velocity", max_velocity_);
    replace_vector6_param(content, "max_acceleration", max_acceleration_);
    replace_vector6_param(content, "max_acceleration_rate",
                          max_acceleration_rate_);

    // Save gravity compensation parameter
    auto replace_scalar_param = [](std::string& content,
                                   const std::string& param, double value) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(1);
      ss << param << ": " << value;

      std::string::size_type start_pos = content.find(param + ": ");
      if (start_pos == std::string::npos) {
        // If parameter not found, add it to the end
        content += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = content.find("\n", start_pos);
        if (end_pos == std::string::npos) {
          end_pos = content.length();
        }
        content.replace(start_pos, end_pos - start_pos, ss.str());
      }
    };

    replace_scalar_param(content, "gravity_compensation_z",
                         gravity_compensation_z_);

    std::ofstream out_file(config_file_);
    if (!out_file.is_open()) {
      ROS_ERROR_STREAM(
          "Failed to open config file for writing: " << config_file_);
      return;
    }
    out_file << content;
    out_file.close();

    ROS_INFO_STREAM("Parameters saved to " << config_file_);
  }

  ros::Rate rate_;
  ros::NodeHandle nh_;
  ros::Subscriber odometry_sub_;
  ros::Subscriber cmd_vel_sub_;
  ros::Subscriber cmd_pose_sub_;
  ros::Subscriber accel_sub_;
  ros::Subscriber dvl_is_valid_sub_;
  ros::Publisher wrench_pub_;
  ros::Publisher desired_velocity_pub_;

  ControlEnableSub control_enable_sub_;
  ControllerBasePtr controller_;
  ros::Time latest_cmd_pose_time_{ros::Time(0)};
  ros::Time latest_cmd_vel_time_{ros::Time(0)};
  ros::Time latest_odometry_time_{ros::Time(0)};
  ros::Time dvl_invalid_since_{ros::Time(0)};
  bool has_dvl_is_valid_message_{false};
  bool dvl_is_valid_{true};
  double dvl_invalid_velocity_zero_timeout_{1.0};

  ControllerBase::StateVector state_{ControllerBase::StateVector::Zero()};
  ControllerBase::StateVector desired_state_{
      ControllerBase::StateVector::Zero()};
  ControllerBase::StateVector d_state_{ControllerBase::StateVector::Zero()};

  ControllerLoader controller_loader_{"auv_controllers",
                                      "auv::control::SixDOFControllerBase"};

  dynamic_reconfigure::Server<auv_control::ControllerConfig>
      dr_srv_;  // Dynamic reconfigure server
  Eigen::Matrix<double, 12, 1> kp_, ki_,
      kd_;  // Parameters to be dynamically reconfigured
  Eigen::Matrix<double, 6, 1> max_velocity_;
  Eigen::Matrix<double, 6, 1> max_acceleration_;
  Eigen::Matrix<double, 6, 1> max_acceleration_rate_;
  Eigen::Vector2d dvl_invalid_position_kp_xy_;
  Eigen::Vector2d dvl_invalid_position_ki_xy_;
  Eigen::Vector2d dvl_invalid_position_kd_xy_;
  Eigen::Vector2d dvl_invalid_velocity_kp_xy_;
  Eigen::Vector2d dvl_invalid_velocity_ki_xy_;
  Eigen::Vector2d dvl_invalid_velocity_kd_xy_;
  Eigen::Matrix<double, 12, 1>
      integral_clamp_limits_;           // Integral clamping limits
  double gravity_compensation_z_{0.0};  // Gravity compensation for z-axis
  std::string config_file_;             // Path to the config file

  std::string depth_control_reference_frame_;
};

}  // namespace control
}  // namespace auv

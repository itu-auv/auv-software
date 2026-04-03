#pragma once

#include <auv_control/ControllerConfig.h>
#include <dynamic_reconfigure/server.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

#include "auv_common_lib/ros/conversions.h"
#include "auv_common_lib/ros/rosparam.h"
#include "auv_common_lib/ros/subscriber_with_timeout.h"
#include "auv_controllers/controller_base.h"
#include "geometry_msgs/AccelWithCovarianceStamped.h"
#include "geometry_msgs/Wrench.h"
#include "nav_msgs/Odometry.h"
#include "pluginlib/class_loader.h"
#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "tf2_ros/buffer.h"

namespace auv {
namespace control {

class ControllerROS {
 public:
  using ControllerBase = SixDOFControllerBase;
  using Model = SixDOFModel;
  using ModelParser = auv::common::rosparam::parser<Model>;
  using Vector6RosparamParser =
      auv::common::rosparam::parser<Eigen::Matrix<double, 6, 1>>;
  using ControllerLoader = pluginlib::ClassLoader<SixDOFControllerBase>;
  using ControllerBasePtr =
      boost::shared_ptr<auv::control::SixDOFControllerBase>;
  using ControlEnableSub =
      auv::common::ros::SubscriberWithTimeout<std_msgs::Bool>;

  explicit ControllerROS(const ros::NodeHandle& nh)
      : nh_{nh},
        rate_{1.0},
        control_enable_sub_{nh},
        tf_buffer_{},
        tf_listener_{tf_buffer_} {
    ros::NodeHandle nh_private("~");

    depth_control_reference_frame_ =
        nh_private.param<std::string>("depth_control_reference_frame", "odom");
    controller_plugin_ = nh_private.param<std::string>(
        "controller_plugin", "auv::control::SixDOFGeometricController");

    const auto model = ModelParser::parse("model", nh_private);
    load_parameters();

    ROS_INFO_STREAM("Model: \n" << model);
    log_parameters();

    const auto rate = nh_private.param("rate", 10.0);
    rate_ = ros::Rate{rate};

    nh_private.param<std::string>("body_frame", body_frame_, "taluy/base_link");
    nh_private.param<double>("transform_timeout", transform_timeout_, 1.0);
    nh_private.param<double>("odometry_timeout", odometry_timeout_, 1.0);

    if (!load_controller(controller_plugin_)) {
      throw std::runtime_error("failed to load controller plugin: " +
                               controller_plugin_);
    }

    controller_->set_model(model);
    apply_controller_parameters();

    auv_control::ControllerConfig initial_config;
    set_initial_config(initial_config);

    dynamic_reconfigure::Server<auv_control::ControllerConfig>::CallbackType f;
    f = boost::bind(&ControllerROS::reconfigure_callback, this, _1, _2);
    dr_srv_.updateConfig(initial_config);
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

    control_enable_sub_.subscribe(
        "enable", 1, nullptr,
        []() { ROS_WARN_STREAM("control enable message timeouted"); },
        ros::Duration{1.0});
    control_enable_sub_.set_default_message(std_msgs::Bool{});

    wrench_pub_ = nh_.advertise<geometry_msgs::WrenchStamped>("wrench", 1);
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
    while (ros::ok()) {
      ros::spinOnce();

      const ros::Time now = ros::Time::now();
      double dt = rate_.expectedCycleTime().toSec();
      if (!last_control_time_.isZero()) {
        dt = (now - last_control_time_).toSec();
      }
      if (dt <= 0.0) {
        dt = rate_.expectedCycleTime().toSec();
      }
      last_control_time_ = now;

      if (!controller_) {
        rate_.sleep();
        continue;
      }

      if ((now - latest_cmd_vel_time_).toSec() > 1.0) {
        desired_command_.twist_ff = SixDOFTwist{};
      }

      const auto control_output =
          controller_->control(state_, desired_command_, state_derivative_, dt);

      geometry_msgs::WrenchStamped wrench_msg;
      if (is_control_enabled() && !is_timeouted() && has_fresh_odometry()) {
        wrench_msg.header.stamp = now;
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

      rate_.sleep();
    }
  }

 private:
  bool is_control_enabled() { return control_enable_sub_.get_message().data; }

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

  void apply_controller_parameters() {
    if (!controller_) {
      return;
    }
    controller_->set_parameters(parameters_);
  }

  void sync_command_pose_to_state() {
    desired_command_.pose = state_.pose;
    desired_command_.normalize_orientation();
  }

  void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    state_ = auv::common::conversions::convert<nav_msgs::Odometry, SixDOFState>(
        *msg);
    latest_odometry_time_ =
        msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
  }

  void cmd_vel_callback(const geometry_msgs::Twist::ConstPtr& msg) {
    if ((ros::Time::now() - latest_cmd_pose_time_).toSec() > 1.0) {
      sync_command_pose_to_state();
    }

    desired_command_.twist_ff =
        auv::common::conversions::convert<geometry_msgs::Twist, SixDOFTwist>(
            *msg);
    latest_cmd_vel_time_ = ros::Time::now();
  }

  std::optional<std::string> get_source_frame(
      const std::string& source_frame) const {
    if (source_frame.empty()) {
      return std::nullopt;
    }

    if (source_frame == depth_control_reference_frame_) {
      return std::nullopt;
    }

    if (source_frame[0] == '/') {
      return source_frame.substr(1);
    }
    return source_frame;
  }

  void cmd_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    const auto source_frame = get_source_frame(msg->header.frame_id);
    auto transformed_pose = msg->pose;

    if (source_frame.has_value()) {
      try {
        const auto transform_stamped = tf_buffer_.lookupTransform(
            depth_control_reference_frame_, source_frame.value(), ros::Time(0),
            ros::Duration(transform_timeout_));

        tf2::doTransform(msg->pose, transformed_pose, transform_stamped);
      } catch (tf2::TransformException& ex) {
        ROS_DEBUG("Failed to transform pose");
        return;
      }
    }

    desired_command_.pose =
        auv::common::conversions::convert<geometry_msgs::Pose, SixDOFPose>(
            transformed_pose);
    latest_cmd_pose_time_ = ros::Time::now();
  }

  void accel_callback(
      const geometry_msgs::AccelWithCovarianceStamped::ConstPtr& msg) {
    state_derivative_ = auv::common::conversions::convert<
        geometry_msgs::Accel, SixDOFStateDerivative>(msg->accel.accel);
  }

  void reconfigure_callback(auv_control::ControllerConfig& config,
                            uint32_t /* level */) {
    parameters_.pose_kp << config.pose_kp_0, config.pose_kp_1, config.pose_kp_2,
        config.pose_kp_3, config.pose_kp_4, config.pose_kp_5;
    parameters_.vel_kp << config.vel_kp_0, config.vel_kp_1, config.vel_kp_2,
        config.vel_kp_3, config.vel_kp_4, config.vel_kp_5;
    parameters_.vel_ki << config.vel_ki_0, config.vel_ki_1, config.vel_ki_2,
        config.vel_ki_3, config.vel_ki_4, config.vel_ki_5;
    parameters_.vel_kd << config.vel_kd_0, config.vel_kd_1, config.vel_kd_2,
        config.vel_kd_3, config.vel_kd_4, config.vel_kd_5;
    parameters_.vel_integral_clamp_limits << config.vel_integral_clamp_limits_0,
        config.vel_integral_clamp_limits_1, config.vel_integral_clamp_limits_2,
        config.vel_integral_clamp_limits_3, config.vel_integral_clamp_limits_4,
        config.vel_integral_clamp_limits_5;
    parameters_.max_velocity << config.max_velocity_0, config.max_velocity_1,
        config.max_velocity_2, config.max_velocity_3, config.max_velocity_4,
        config.max_velocity_5;
    parameters_.gravity_compensation_z = config.gravity_compensation_z;

    apply_controller_parameters();
    save_parameters();
  }

  void load_parameters() {
    ros::NodeHandle nh_private("~");

    if (nh_private.hasParam("pose_kp")) {
      parameters_.pose_kp = Vector6RosparamParser::parse("pose_kp", nh_private);
    } else {
      parameters_.pose_kp = ControllerBase::Vector::Zero();
    }

    if (nh_private.hasParam("vel_kp")) {
      parameters_.vel_kp = Vector6RosparamParser::parse("vel_kp", nh_private);
    } else {
      parameters_.vel_kp = ControllerBase::Vector::Zero();
    }

    if (nh_private.hasParam("vel_ki")) {
      parameters_.vel_ki = Vector6RosparamParser::parse("vel_ki", nh_private);
    } else {
      parameters_.vel_ki = ControllerBase::Vector::Zero();
    }

    if (nh_private.hasParam("vel_kd")) {
      parameters_.vel_kd = Vector6RosparamParser::parse("vel_kd", nh_private);
    } else {
      parameters_.vel_kd = ControllerBase::Vector::Zero();
    }

    if (nh_private.hasParam("vel_integral_clamp_limits")) {
      parameters_.vel_integral_clamp_limits =
          Vector6RosparamParser::parse("vel_integral_clamp_limits", nh_private);
    } else {
      parameters_.vel_integral_clamp_limits = ControllerBase::Vector::Zero();
    }

    parameters_.gravity_compensation_z =
        nh_private.param("gravity_compensation_z", 0.0);

    if (nh_private.hasParam("max_velocity")) {
      parameters_.max_velocity =
          Vector6RosparamParser::parse("max_velocity", nh_private);
    } else {
      parameters_.max_velocity = ControllerBase::Vector::Constant(1e6);
    }
  }

  void set_initial_config(auv_control::ControllerConfig& config) {
    config.pose_kp_0 = parameters_.pose_kp(0);
    config.pose_kp_1 = parameters_.pose_kp(1);
    config.pose_kp_2 = parameters_.pose_kp(2);
    config.pose_kp_3 = parameters_.pose_kp(3);
    config.pose_kp_4 = parameters_.pose_kp(4);
    config.pose_kp_5 = parameters_.pose_kp(5);

    config.vel_kp_0 = parameters_.vel_kp(0);
    config.vel_kp_1 = parameters_.vel_kp(1);
    config.vel_kp_2 = parameters_.vel_kp(2);
    config.vel_kp_3 = parameters_.vel_kp(3);
    config.vel_kp_4 = parameters_.vel_kp(4);
    config.vel_kp_5 = parameters_.vel_kp(5);

    config.vel_ki_0 = parameters_.vel_ki(0);
    config.vel_ki_1 = parameters_.vel_ki(1);
    config.vel_ki_2 = parameters_.vel_ki(2);
    config.vel_ki_3 = parameters_.vel_ki(3);
    config.vel_ki_4 = parameters_.vel_ki(4);
    config.vel_ki_5 = parameters_.vel_ki(5);

    config.vel_kd_0 = parameters_.vel_kd(0);
    config.vel_kd_1 = parameters_.vel_kd(1);
    config.vel_kd_2 = parameters_.vel_kd(2);
    config.vel_kd_3 = parameters_.vel_kd(3);
    config.vel_kd_4 = parameters_.vel_kd(4);
    config.vel_kd_5 = parameters_.vel_kd(5);

    config.vel_integral_clamp_limits_0 =
        parameters_.vel_integral_clamp_limits(0);
    config.vel_integral_clamp_limits_1 =
        parameters_.vel_integral_clamp_limits(1);
    config.vel_integral_clamp_limits_2 =
        parameters_.vel_integral_clamp_limits(2);
    config.vel_integral_clamp_limits_3 =
        parameters_.vel_integral_clamp_limits(3);
    config.vel_integral_clamp_limits_4 =
        parameters_.vel_integral_clamp_limits(4);
    config.vel_integral_clamp_limits_5 =
        parameters_.vel_integral_clamp_limits(5);

    config.gravity_compensation_z = parameters_.gravity_compensation_z;
    config.max_velocity_0 = parameters_.max_velocity(0);
    config.max_velocity_1 = parameters_.max_velocity(1);
    config.max_velocity_2 = parameters_.max_velocity(2);
    config.max_velocity_3 = parameters_.max_velocity(3);
    config.max_velocity_4 = parameters_.max_velocity(4);
    config.max_velocity_5 = parameters_.max_velocity(5);
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

    auto replace_vector6_param = [](std::string& yaml, const std::string& param,
                                    const ControllerBase::Vector& values) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(4);
      ss << param << ": [" << values(0);
      for (int i = 1; i < values.size(); ++i) {
        ss << ", " << values(i);
      }
      ss << "]";

      std::string::size_type start_pos = yaml.find(param + ": [");
      if (start_pos == std::string::npos) {
        yaml += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = yaml.find("]", start_pos);
        yaml.replace(start_pos, end_pos - start_pos + 1, ss.str());
      }
    };

    auto replace_scalar_param = [](std::string& yaml, const std::string& param,
                                   double value) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(4);
      ss << param << ": " << value;

      std::string::size_type start_pos = yaml.find(param + ": ");
      if (start_pos == std::string::npos) {
        yaml += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = yaml.find("\n", start_pos);
        if (end_pos == std::string::npos) {
          end_pos = yaml.length();
        }
        yaml.replace(start_pos, end_pos - start_pos, ss.str());
      }
    };

    auto replace_string_param = [](std::string& yaml, const std::string& param,
                                   const std::string& value) {
      std::stringstream ss;
      ss << param << ": \"" << value << "\"";

      std::string::size_type start_pos = yaml.find(param + ": ");
      if (start_pos == std::string::npos) {
        yaml += "\n" + ss.str();
      } else {
        std::string::size_type end_pos = yaml.find("\n", start_pos);
        if (end_pos == std::string::npos) {
          end_pos = yaml.length();
        }
        yaml.replace(start_pos, end_pos - start_pos, ss.str());
      }
    };

    replace_string_param(content, "controller_plugin", controller_plugin_);
    replace_vector6_param(content, "pose_kp", parameters_.pose_kp);
    replace_vector6_param(content, "vel_kp", parameters_.vel_kp);
    replace_vector6_param(content, "vel_ki", parameters_.vel_ki);
    replace_vector6_param(content, "vel_kd", parameters_.vel_kd);
    replace_vector6_param(content, "vel_integral_clamp_limits",
                          parameters_.vel_integral_clamp_limits);
    replace_vector6_param(content, "max_velocity", parameters_.max_velocity);
    replace_scalar_param(content, "gravity_compensation_z",
                         parameters_.gravity_compensation_z);

    std::ofstream out_file(config_file_);
    if (!out_file.is_open()) {
      ROS_ERROR_STREAM(
          "Failed to open config file for writing: " << config_file_);
      return;
    }
    out_file << content;
    out_file.close();
  }

  void log_parameters() const {
    ROS_INFO_STREAM("controller_plugin: " << controller_plugin_);
    ROS_INFO_STREAM("pose_kp: \n" << parameters_.pose_kp.transpose());
    ROS_INFO_STREAM("vel_kp: \n" << parameters_.vel_kp.transpose());
    ROS_INFO_STREAM("vel_ki: \n" << parameters_.vel_ki.transpose());
    ROS_INFO_STREAM("vel_kd: \n" << parameters_.vel_kd.transpose());
    ROS_INFO_STREAM("vel_integral_clamp_limits: \n"
                    << parameters_.vel_integral_clamp_limits.transpose());
    ROS_INFO_STREAM(
        "gravity_compensation_z: " << parameters_.gravity_compensation_z);
    ROS_INFO_STREAM("max_velocity: \n" << parameters_.max_velocity.transpose());
  }

  ros::Rate rate_;
  ros::NodeHandle nh_;
  ros::Subscriber odometry_sub_;
  ros::Subscriber cmd_vel_sub_;
  ros::Subscriber cmd_pose_sub_;
  ros::Subscriber accel_sub_;
  ros::Publisher wrench_pub_;

  ControlEnableSub control_enable_sub_;
  ControllerBasePtr controller_;
  ros::Time latest_cmd_pose_time_{ros::Time(0)};
  ros::Time latest_cmd_vel_time_{ros::Time(0)};
  ros::Time latest_odometry_time_{ros::Time(0)};
  ros::Time last_control_time_{ros::Time(0)};

  SixDOFState state_{};
  SixDOFCommand desired_command_{};
  SixDOFStateDerivative state_derivative_{};

  ControllerLoader controller_loader_{"auv_controllers",
                                      "auv::control::SixDOFControllerBase"};

  dynamic_reconfigure::Server<auv_control::ControllerConfig> dr_srv_;
  SixDOFControllerParameters parameters_{};
  std::string config_file_;
  std::string controller_plugin_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string body_frame_;
  double transform_timeout_{1.0};
  double odometry_timeout_{1.0};
  std::string depth_control_reference_frame_;
};

}  // namespace control
}  // namespace auv

#pragma once

#include <auv_control/ControllerConfig.h>  // Include your dynamic reconfigure header
#include <dynamic_reconfigure/server.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <type_traits>

#include "auv_common_lib/ros/conversions.h"
#include "auv_common_lib/ros/rosparam.h"
#include "auv_common_lib/ros/subscriber_with_timeout.h"
#include "auv_controllers/controller_base.h"
#include "auv_controllers/multidof_pid_controller.h"
#include "geometry_msgs/AccelWithCovarianceStamped.h"
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

    ROS_INFO_STREAM("kp: \n" << kp_.transpose());
    ROS_INFO_STREAM("ki: \n" << ki_.transpose());
    ROS_INFO_STREAM("kd: \n" << kd_.transpose());
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

    // Set rigid body dynamics parameters
    controller->set_enable_rigid_body_coriolis(enable_rigid_body_coriolis_);
    controller->set_enable_restoring(enable_restoring_);
    if (enable_restoring_) {
      controller->set_hydrostatic_params(hydro_mass_, hydro_displaced_volume_,
                                         hydro_fluid_density_, hydro_gravity_,
                                         r_g_body_, r_b_body_);
    }

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

      const auto control_output =
          controller_->control(state_, desired_state_, d_state_, dt);

      geometry_msgs::WrenchStamped wrench_msg;
      if (is_control_enabled() && !is_timeouted()) {
        wrench_msg.header.stamp = ros::Time::now();
        wrench_msg.header.frame_id = body_frame_;
        wrench_msg.wrench =
            auv::common::conversions::convert<ControllerBase::WrenchVector,
                                              geometry_msgs::Wrench>(
                control_output);
        wrench_pub_.publish(wrench_msg);
      }
    }
  }

 private:
  bool is_control_enabled() { return control_enable_sub_.get_message().data; }
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string body_frame_;
  double transform_timeout_;

  bool is_timeouted() const {
    const auto latest_time =
        std::max(latest_cmd_vel_time_, latest_cmd_pose_time_);
    return (ros::Time::now() - latest_time).toSec() > 1.0;
  }

  void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    state_ =
        auv::common::conversions::convert<nav_msgs::Odometry,
                                          ControllerBase::StateVector>(*msg);
    d_state_.head(6) = state_.tail(6);
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

    static tf2_ros::Buffer tf_buffer;
    static tf2_ros::TransformListener tf_listener(tf_buffer);
    geometry_msgs::TransformStamped transform_stamped;

    if (source_frame.has_value()) {
      ROS_DEBUG("Source frame: %s, Desired frame: %s",
                source_frame.value().c_str(),
                depth_control_reference_frame_.c_str());
      try {
        transform_stamped = tf_buffer.lookupTransform(
            depth_control_reference_frame_, source_frame.value(), ros::Time(0),
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

    latest_cmd_pose_time_ = ros::Time::now();
  }

  void accel_callback(
      const geometry_msgs::AccelWithCovarianceStamped::ConstPtr& msg) {
    d_state_(6) = msg->accel.accel.linear.x;
    d_state_(7) = msg->accel.accel.linear.y;
    d_state_(8) = msg->accel.accel.linear.z;
    d_state_.tail(3) = Eigen::Vector3d::Zero();
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
    controller->set_kp(kp_);
    controller->set_ki(ki_);
    controller->set_kd(kd_);
    controller->set_integral_clamp_limits(integral_clamp_limits_);
    controller->set_gravity_compensation_z(config.gravity_compensation_z);
    gravity_compensation_z_ = config.gravity_compensation_z;

    max_velocity_ << config.max_velocity_0, config.max_velocity_1,
        config.max_velocity_2, config.max_velocity_3, config.max_velocity_4,
        config.max_velocity_5;
    controller->set_max_velocity_limits(max_velocity_);

    save_parameters();
  }

  void load_parameters() {
    kp_ = VectorRosparamParser::parse("kp", ros::NodeHandle("~"));
    ki_ = VectorRosparamParser::parse("ki", ros::NodeHandle("~"));
    kd_ = VectorRosparamParser::parse("kd", ros::NodeHandle("~"));

    // Load integral clamp limits with default values of 0 (0 means no clamping)
    ros::NodeHandle nh_private("~");
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

    // Load rigid body dynamics parameters
    enable_rigid_body_coriolis_ =
        nh_private.param("enable_rigid_body_coriolis", false);
    enable_restoring_ = nh_private.param("enable_restoring", false);
    ROS_INFO_STREAM("Rigid body coriolis: "
                    << (enable_rigid_body_coriolis_ ? "enabled" : "disabled"));
    ROS_INFO_STREAM("Hydrostatic restoring: "
                    << (enable_restoring_ ? "enabled" : "disabled"));

    // Load hydrostatic parameters if restoring is enabled
    if (enable_restoring_) {
      hydro_mass_ = nh_private.param("hydrostatic/mass", 0.0);
      hydro_displaced_volume_ =
          nh_private.param("hydrostatic/displaced_volume", 0.0);
      hydro_fluid_density_ =
          nh_private.param("hydrostatic/fluid_density", 1000.0);
      hydro_gravity_ = nh_private.param("hydrostatic/gravity", 9.80665);

      std::vector<double> r_g_default = {0.0, 0.0, 0.0};
      std::vector<double> r_b_default = {0.0, 0.0, 0.0};
      std::vector<double> r_g_vec, r_b_vec;
      nh_private.param("hydrostatic/r_g_body", r_g_vec, r_g_default);
      nh_private.param("hydrostatic/r_b_body", r_b_vec, r_b_default);

      if (r_g_vec.size() == 3 && r_b_vec.size() == 3) {
        r_g_body_ = Eigen::Vector3d(r_g_vec[0], r_g_vec[1], r_g_vec[2]);
        r_b_body_ = Eigen::Vector3d(r_b_vec[0], r_b_vec[1], r_b_vec[2]);
      } else {
        ROS_WARN("Invalid r_g_body or r_b_body size, using zeros");
        r_g_body_ = Eigen::Vector3d::Zero();
        r_b_body_ = Eigen::Vector3d::Zero();
      }

      ROS_INFO_STREAM("Hydrostatic params - mass: "
                      << hydro_mass_
                      << ", displaced_volume: " << hydro_displaced_volume_
                      << ", fluid_density: " << hydro_fluid_density_);
      ROS_INFO_STREAM("r_g_body: " << r_g_body_.transpose());
      ROS_INFO_STREAM("r_b_body: " << r_b_body_.transpose());
    }
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
  ros::Publisher wrench_pub_;

  ControlEnableSub control_enable_sub_;
  ControllerBasePtr controller_;
  ros::Time latest_cmd_pose_time_{ros::Time(0)};
  ros::Time latest_cmd_vel_time_{ros::Time(0)};

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
  Eigen::Matrix<double, 12, 1>
      integral_clamp_limits_;           // Integral clamping limits
  double gravity_compensation_z_{0.0};  // Gravity compensation for z-axis
  std::string config_file_;             // Path to the config file

  std::string depth_control_reference_frame_;

  // Rigid body dynamics parameters
  bool enable_rigid_body_coriolis_{false};
  bool enable_restoring_{false};
  double hydro_mass_{0.0};
  double hydro_displaced_volume_{0.0};
  double hydro_fluid_density_{1000.0};
  double hydro_gravity_{9.80665};
  Eigen::Vector3d r_g_body_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d r_b_body_{Eigen::Vector3d::Zero()};
};

}  // namespace control
}  // namespace auv

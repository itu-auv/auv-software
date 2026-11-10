#pragma once

#include <tf2_ros/transform_listener.h>

#include <algorithm>
#include <cmath>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <type_traits>

#include "auv_common_lib/ros/conversions.h"
#include "auv_common_lib/ros/rosparam.h"
#include "auv_common_lib/ros/subscriber_with_timeout.h"
#include "auv_controllers/controller_base.h"
#include "auv_controllers/multidof_pid_controller.h"
#include "geometry_msgs/msg/accel_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/wrench.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
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
  using ControllerBasePtr = std::shared_ptr<auv::control::SixDOFControllerBase>;
  using ControlEnableSub =
      auv::common::ros::SubscriberWithTimeout<std_msgs::msg::Bool>;

  ControllerROS(const rclcpp::Node::SharedPtr& nh)
      : nh_{nh},
        rate_{1.0},
        control_enable_sub_{nh},
        tf_buffer_{nh->get_clock()},
        tf_listener_{tf_buffer_} {
    auto& nh_private = *nh_;

    // default_frame as a ros parameter (default is odom)
    depth_control_reference_frame_ = nh_private.declare_parameter<std::string>(
        "depth_control_reference_frame", "odom");

    for (const auto& key : {"mass_inertia_matrix", "linear_damping_matrix",
                            "quadratic_damping_matrix"})
      nh_->declare_parameter<std::vector<double>>(std::string("model.") + key);
    auto model = ModelParser::parse("model", nh_private);
    load_parameters();

    RCLCPP_INFO_STREAM(nh_->get_logger(), "Model: \n" << model);

    const auto rate = nh_private.declare_parameter<double>("rate", 10.0);
    rate_ = rate;

    body_frame_ = nh_private.declare_parameter<std::string>("body_frame",
                                                            "taluy/base_link");
    transform_timeout_ =
        nh_private.declare_parameter<double>("transform_timeout", 1.0);
    odometry_timeout_ =
        nh_private.declare_parameter<double>("odometry_timeout", 1.0);

    RCLCPP_INFO_STREAM(nh_->get_logger(), "kp: \n" << kp_.transpose());
    RCLCPP_INFO_STREAM(nh_->get_logger(), "ki: \n" << ki_.transpose());
    RCLCPP_INFO_STREAM(nh_->get_logger(), "kd: \n" << kd_.transpose());
    RCLCPP_INFO_STREAM(nh_->get_logger(),
                       "integral_clamp_limits: \n"
                           << integral_clamp_limits_.transpose());
    RCLCPP_INFO_STREAM(nh_->get_logger(),
                       "gravity_compensation_z: " << gravity_compensation_z_);
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

    parameter_callback_ = nh_->add_on_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& params) {
          return validate_controller_parameters(params);
        });
    parameter_applied_callback_ = nh_->add_post_set_parameters_callback(
        [this](const std::vector<rclcpp::Parameter>& params) {
          if (std::any_of(params.begin(), params.end(), [](const auto& param) {
                return is_controller_parameter(param.get_name());
              })) {
            apply_controller_parameters();
          }
        });
    odometry_sub_ = nh_->create_subscription<nav_msgs::msg::Odometry>(
        "odometry", 1,
        std::bind(&ControllerROS::odometry_callback, this,
                  std::placeholders::_1));
    cmd_vel_sub_ = nh_->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 1,
        std::bind(&ControllerROS::cmd_vel_callback, this,
                  std::placeholders::_1));
    cmd_pose_sub_ = nh_->create_subscription<geometry_msgs::msg::PoseStamped>(
        "cmd_pose", 1,
        std::bind(&ControllerROS::cmd_pose_callback, this,
                  std::placeholders::_1));
    accel_sub_ = nh_->create_subscription<
        geometry_msgs::msg::AccelWithCovarianceStamped>(
        "acceleration", 1,
        std::bind(&ControllerROS::accel_callback, this, std::placeholders::_1));
    cmd_wrench_sub_ =
        nh_->create_subscription<geometry_msgs::msg::WrenchStamped>(
            "cmd_wrench", 1,
            std::bind(&ControllerROS::cmd_wrench_callback, this,
                      std::placeholders::_1));
    control_enable_sub_.subscribe(
        "enable", 1, nullptr,
        [this]() {
          RCLCPP_WARN(nh_->get_logger(), "control enable message timeouted");
        },
        rclcpp::Duration::from_seconds(1.0));
    control_enable_sub_.set_default_message(std_msgs::msg::Bool{});
    wrench_pub_ =
        nh_->create_publisher<geometry_msgs::msg::WrenchStamped>("wrench", 1);
    desired_velocity_pub_ =
        nh_->create_publisher<geometry_msgs::msg::Twist>("desired_velocity", 1);
  }

  bool load_controller(const std::string& controller_name) {
    try {
      controller_ = controller_loader_.createSharedInstance(controller_name);
      RCLCPP_INFO_STREAM(nh_->get_logger(),
                         "Controller loaded: " << controller_name);

      return true;
    } catch (pluginlib::PluginlibException& ex) {
      RCLCPP_ERROR(nh_->get_logger(),
                   "The plugin failed to load for some reason. Error: %s",
                   ex.what());
      return false;
    }
  }

  void spin() {
    // Nominal control period in seconds, matching the timer period.
    const auto dt = 1.0 / rate_;
    auto timer = rclcpp::create_timer(
        nh_, nh_->get_clock(), rclcpp::Duration::from_seconds(dt),
        [this, dt]() {
          if (!controller_) {
            return;
          }

          if ((nh_->now() - latest_cmd_vel_time_).seconds() > 1.0) {
            desired_state_.tail(6) = ControllerBase::Vector::Zero();
          }

          auto controller = dynamic_cast<auv::control::SixDOFPIDController*>(
              controller_.get());
          if (controller) {
            controller->set_external_wrench(
                has_recent_cmd_wrench() ? cmd_wrench_
                                        : ControllerBase::WrenchVector::Zero());
          }
          const auto control_output =
              controller_->control(state_, desired_state_, d_state_, dt);

          auto pid_controller =
              dynamic_cast<auv::control::SixDOFPIDController*>(
                  controller_.get());
          desired_velocity_pub_->publish(
              auv::common::conversions::convert<ControllerBase::Vector,
                                                geometry_msgs::msg::Twist>(
                  pid_controller->get_desired_velocity()));

          geometry_msgs::msg::WrenchStamped wrench_msg;
          if (is_control_enabled() && !is_timeouted() && has_fresh_odometry()) {
            wrench_msg.header.stamp = nh_->now();
            wrench_msg.header.frame_id = body_frame_;
            wrench_msg.wrench =
                auv::common::conversions::convert<ControllerBase::WrenchVector,
                                                  geometry_msgs::msg::Wrench>(
                    control_output);
            wrench_pub_->publish(wrench_msg);
          } else if (is_control_enabled() && !has_fresh_odometry()) {
            RCLCPP_WARN_THROTTLE(
                nh_->get_logger(), *nh_->get_clock(), 3000,
                "control enable requested but odometry is missing or "
                "stale, not publishing wrench");
          }
        });
    rclcpp::spin(nh_);
  }

 private:
  bool is_control_enabled() { return control_enable_sub_.get_message().data; }
  bool has_recent_cmd_wrench() const {
    return (nh_->now() - latest_cmd_wrench_time_).seconds() <=
           cmd_wrench_timeout_;
  }
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::string body_frame_;
  double transform_timeout_;
  double cmd_wrench_timeout_{0.5};
  double odometry_timeout_;
  bool is_timeouted() const {
    const auto latest_time =
        std::max(latest_cmd_vel_time_, latest_cmd_pose_time_);
    return (nh_->now() - latest_time).seconds() > 1.0;
  }

  bool has_fresh_odometry() const {
    if (latest_odometry_time_.nanoseconds() == 0) {
      return false;
    }

    return (nh_->now() - latest_odometry_time_).seconds() <= odometry_timeout_;
  }

  void odometry_callback(const nav_msgs::msg::Odometry::ConstSharedPtr& msg) {
    state_ =
        auv::common::conversions::convert<nav_msgs::msg::Odometry,
                                          ControllerBase::StateVector>(*msg);
    d_state_.head(6) = state_.tail(6);
    latest_odometry_time_ =
        rclcpp::Time(msg->header.stamp, nh_->get_clock()->get_clock_type())
                    .nanoseconds() == 0
            ? nh_->now()
            : rclcpp::Time(msg->header.stamp,
                           nh_->get_clock()->get_clock_type());
  }

  void cmd_vel_callback(const geometry_msgs::msg::Twist::ConstSharedPtr& msg) {
    if ((nh_->now() - latest_cmd_pose_time_).seconds() > 1.0) {
      desired_state_.head(6) = state_.head(6);
    }

    desired_state_.tail(6) =
        auv::common::conversions::convert<geometry_msgs::msg::Twist,
                                          ControllerBase::Vector>(*msg);
    latest_cmd_vel_time_ = nh_->now();
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

  void cmd_pose_callback(
      const geometry_msgs::msg::PoseStamped::ConstSharedPtr& msg) {
    const auto source_frame = get_source_frame(msg->header.frame_id);
    auto transformed_pose = msg->pose;
    const auto lookup_time =
        rclcpp::Time(msg->header.stamp, nh_->get_clock()->get_clock_type());

    geometry_msgs::msg::TransformStamped transform_stamped;

    if (source_frame.has_value()) {
      RCLCPP_DEBUG(nh_->get_logger(), "Source frame: %s, Desired frame: %s",
                   source_frame.value().c_str(),
                   depth_control_reference_frame_.c_str());
      try {
        transform_stamped = tf_buffer_.lookupTransform(
            depth_control_reference_frame_, source_frame.value(), lookup_time,
            rclcpp::Duration::from_seconds(transform_timeout_));

        tf2::doTransform(msg->pose, transformed_pose, transform_stamped);
      } catch (tf2::TransformException& ex) {
        RCLCPP_DEBUG(nh_->get_logger(), "Failed to transform pose");
        return;
      }
    }
    RCLCPP_DEBUG_STREAM(nh_->get_logger(),
                        "Final transformed command pose: "
                            << transformed_pose.position.x << ", "
                            << transformed_pose.position.y << ", "
                            << transformed_pose.position.z);

    desired_state_.head(6) = auv::common::conversions::convert<
        geometry_msgs::msg::Pose, ControllerBase::Vector>(transformed_pose);

    latest_cmd_pose_time_ =
        lookup_time.nanoseconds() == 0 ? nh_->now() : lookup_time;
  }

  void accel_callback(
      const geometry_msgs::msg::AccelWithCovarianceStamped::ConstSharedPtr&
          msg) {
    d_state_(6) = msg->accel.accel.linear.x;
    d_state_(7) = msg->accel.accel.linear.y;
    d_state_(8) = msg->accel.accel.linear.z;
    d_state_.tail(3) = Eigen::Vector3d::Zero();
  }

  void cmd_wrench_callback(
      const geometry_msgs::msg::WrenchStamped::ConstSharedPtr& msg) {
    cmd_wrench_ << msg->wrench.force.x, msg->wrench.force.y,
        msg->wrench.force.z, msg->wrench.torque.x, msg->wrench.torque.y,
        msg->wrench.torque.z;
    latest_cmd_wrench_time_ = nh_->now();
  }

  static bool is_controller_parameter(const std::string& name) {
    return name == "kp" || name == "ki" || name == "kd" ||
           name == "integral_clamp_limits" || name == "max_velocity" ||
           name == "max_acceleration" || name == "max_acceleration_rate" ||
           name == "gravity_compensation_z";
  }

  rcl_interfaces::msg::SetParametersResult validate_controller_parameters(
      const std::vector<rclcpp::Parameter>& params) const {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    try {
      for (const auto& param : params) {
        const auto& name = param.get_name();
        if (!is_controller_parameter(name)) continue;
        const bool gain = name == "kp" || name == "ki" || name == "kd";
        const bool gravity = name == "gravity_compensation_z";
        const bool clamp = name == "integral_clamp_limits";
        const double low = gain ? -20.0 : gravity ? -100.0 : 0.0;
        const double high = gain                         ? 20.0
                            : (gravity || clamp)         ? 100.0
                            : name == "max_velocity"     ? 5.0
                            : name == "max_acceleration" ? 2.0
                                                         : 1.0;
        std::vector<double> values;
        if (gravity) {
          values = {param.as_double()};
        } else {
          values = param.as_double_array();
          const size_t expected = (gain || clamp) ? 12 : 6;
          if (values.size() != expected) {
            result.successful = false;
            result.reason =
                name + " must contain " + std::to_string(expected) + " doubles";
            return result;
          }
        }
        if (std::any_of(
                values.begin(), values.end(), [low, high](double value) {
                  return !std::isfinite(value) || value < low || value > high;
                })) {
          result.successful = false;
          result.reason = "parameter out of range: " + name;
          return result;
        }
      }
    } catch (const std::exception& ex) {
      result.successful = false;
      result.reason = ex.what();
    }
    return result;
  }

  void apply_controller_parameters() {
    kp_ = VectorRosparamParser::parse("kp", *nh_);
    ki_ = VectorRosparamParser::parse("ki", *nh_);
    kd_ = VectorRosparamParser::parse("kd", *nh_);
    integral_clamp_limits_ =
        VectorRosparamParser::parse("integral_clamp_limits", *nh_);
    max_velocity_ = Vector6RosparamParser::parse("max_velocity", *nh_);
    max_acceleration_ = Vector6RosparamParser::parse("max_acceleration", *nh_);
    max_acceleration_rate_ =
        Vector6RosparamParser::parse("max_acceleration_rate", *nh_);
    gravity_compensation_z_ =
        nh_->get_parameter("gravity_compensation_z").as_double();

    auto controller = dynamic_cast<SixDOFPIDController*>(controller_.get());
    controller->set_kp(kp_);
    controller->set_ki(ki_);
    controller->set_kd(kd_);
    controller->set_integral_clamp_limits(integral_clamp_limits_);
    controller->set_max_velocity_limits(max_velocity_);
    controller->set_max_acceleration_limits(max_acceleration_);
    controller->set_max_acceleration_rate_limits(max_acceleration_rate_);
    controller->set_gravity_compensation_z(gravity_compensation_z_);
  }

  void load_parameters() {
    for (const auto& name : {"kp", "ki", "kd"})
      nh_->declare_parameter<std::vector<double>>(name);
    nh_->declare_parameter<std::vector<double>>("integral_clamp_limits",
                                                std::vector<double>(12, 0.0));
    nh_->declare_parameter<std::vector<double>>("max_velocity",
                                                std::vector<double>(6, 1e6));
    nh_->declare_parameter<std::vector<double>>("max_acceleration",
                                                std::vector<double>(6, 0.0));
    nh_->declare_parameter<std::vector<double>>("max_acceleration_rate",
                                                std::vector<double>(6, 0.0));
    kp_ = VectorRosparamParser::parse("kp", *nh_);
    ki_ = VectorRosparamParser::parse("ki", *nh_);
    kd_ = VectorRosparamParser::parse("kd", *nh_);

    // Load integral clamp limits with default values of 0 (0 means no clamping)
    auto& nh_private = *nh_;
    if (nh_private.has_parameter("integral_clamp_limits")) {
      integral_clamp_limits_ =
          VectorRosparamParser::parse("integral_clamp_limits", *nh_);
      RCLCPP_INFO(nh_->get_logger(), "Loaded integral_clamp_limits parameter");
    } else {
      // If parameter doesn't exist, no clamping
      integral_clamp_limits_ = Eigen::Matrix<double, 12, 1>::Zero();
      RCLCPP_INFO(nh_->get_logger(),
                  "No integral_clamp_limits parameter found, integral clamping "
                  "disabled");
    }

    // Load gravity compensation parameter
    gravity_compensation_z_ =
        nh_private.declare_parameter<double>("gravity_compensation_z", 0.0);

    // Load max velocity limits
    if (nh_private.has_parameter("max_velocity")) {
      max_velocity_ = Vector6RosparamParser::parse("max_velocity", nh_private);
      RCLCPP_INFO_STREAM(nh_->get_logger(),
                         "Loaded max_velocity: " << max_velocity_.transpose());
    } else {
      max_velocity_ = Eigen::Matrix<double, 6, 1>::Constant(1e6);
      RCLCPP_WARN_STREAM(nh_->get_logger(),
                         "No max_velocity parameter found, limits disabled");
    }

    // Load max acceleration command limits
    if (nh_private.has_parameter("max_acceleration")) {
      max_acceleration_ =
          Vector6RosparamParser::parse("max_acceleration", nh_private);
      RCLCPP_INFO_STREAM(
          nh_->get_logger(),
          "Loaded max_acceleration: " << max_acceleration_.transpose());
    } else {
      max_acceleration_ = Eigen::Matrix<double, 6, 1>::Zero();
      RCLCPP_INFO_STREAM(
          nh_->get_logger(),
          "No max_acceleration parameter found, acceleration limits disabled");
    }

    // Load max acceleration command rate limits
    if (nh_private.has_parameter("max_acceleration_rate")) {
      max_acceleration_rate_ =
          Vector6RosparamParser::parse("max_acceleration_rate", nh_private);
      RCLCPP_INFO_STREAM(nh_->get_logger(),
                         "Loaded max_acceleration_rate: "
                             << max_acceleration_rate_.transpose());
    } else {
      max_acceleration_rate_ = Eigen::Matrix<double, 6, 1>::Zero();
      RCLCPP_INFO_STREAM(
          nh_->get_logger(),
          "No max_acceleration_rate parameter found, acceleration rate limits "
          "disabled");
    }
  }

  double rate_;
  rclcpp::Node::SharedPtr nh_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr
      cmd_pose_sub_;
  rclcpp::Subscription<
      geometry_msgs::msg::AccelWithCovarianceStamped>::SharedPtr accel_sub_;
  rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr
      cmd_wrench_sub_;
  rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr wrench_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr desired_velocity_pub_;

  ControlEnableSub control_enable_sub_;
  ControllerLoader controller_loader_{"auv_controllers",
                                      "auv::control::SixDOFControllerBase"};
  ControllerBasePtr controller_;
  rclcpp::Time latest_cmd_pose_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time latest_cmd_vel_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time latest_cmd_wrench_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time latest_odometry_time_{0, 0, RCL_ROS_TIME};

  ControllerBase::StateVector state_{ControllerBase::StateVector::Zero()};
  ControllerBase::StateVector desired_state_{
      ControllerBase::StateVector::Zero()};
  ControllerBase::StateVector d_state_{ControllerBase::StateVector::Zero()};

  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr
      parameter_callback_;
  rclcpp::node_interfaces::PostSetParametersCallbackHandle::SharedPtr
      parameter_applied_callback_;
  Eigen::Matrix<double, 12, 1> kp_, ki_,
      kd_;  // Parameters to be dynamically reconfigured
  Eigen::Matrix<double, 6, 1> max_velocity_;
  ControllerBase::WrenchVector cmd_wrench_{
      ControllerBase::WrenchVector::Zero()};
  Eigen::Matrix<double, 6, 1> max_acceleration_;
  Eigen::Matrix<double, 6, 1> max_acceleration_rate_;
  Eigen::Matrix<double, 12, 1>
      integral_clamp_limits_;           // Integral clamping limits
  double gravity_compensation_z_{0.0};  // Gravity compensation for z-axis

  std::string depth_control_reference_frame_;
};

}  // namespace control
}  // namespace auv

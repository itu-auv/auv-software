#pragma once

#include <type_traits>

#include "auv_common_lib/ros/conversions.h"
#include "auv_common_lib/ros/rosparam.h"
#include "auv_common_lib/ros/subscriber_with_timeout.h"
#include "auv_controllers/controller_base.h"
#include "auv_controllers/multidof_pid_controller.h"
#include "geometry_msgs/Wrench.h"
#include "nav_msgs/Odometry.h"
#include "pluginlib/class_loader.h"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "std_msgs/Bool.h"

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
  using ControllerLoader = pluginlib::ClassLoader<SixDOFControllerBase>;
  using ControllerBasePtr =
      boost::shared_ptr<auv::control::SixDOFControllerBase>;
  using ControlEnableSub =
      auv::common::ros::SubscriberWithTimeout<std_msgs::Bool>;

  ControllerROS(const ros::NodeHandle& nh)
      : nh_{nh}, rate_{1.0}, control_enable_sub_{nh} {
    ros::NodeHandle nh_private("~");

    auto model = ModelParser::parse("model", nh_private);

    const auto kp = VectorRosparamParser::parse("kp", nh_private);

    const auto ki = VectorRosparamParser::parse("ki", nh_private);

    const auto kd = VectorRosparamParser::parse("kd", nh_private);

    ROS_INFO_STREAM("Model: \n" << model);

    const auto rate = nh_private.param("rate", 10.0);
    rate_ = ros::Rate{rate};

    ROS_INFO_STREAM("kp: \n" << kp.transpose());
    ROS_INFO_STREAM("ki: \n" << ki.transpose());
    ROS_INFO_STREAM("kd: \n" << kd.transpose());
    load_controller("auv::control::SixDOFPIDController");

    auto controller =
        dynamic_cast<auv::control::SixDOFPIDController*>(controller_.get());

    controller->set_model(model);
    controller->set_kp(kp);
    controller->set_ki(ki);
    controller->set_kd(kd);

    odometry_sub_ =
        nh_.subscribe("odometry", 1, &ControllerROS::odometry_callback, this);
    cmd_vel_sub_ =
        nh_.subscribe("cmd_vel", 1, &ControllerROS::cmd_vel_callback, this);
    cmd_pose_sub_ =
        nh_.subscribe("cmd_pose", 1, &ControllerROS::cmd_pose_callback, this);
    imu_sub_ = nh_.subscribe("imu", 1, &ControllerROS::imu_callback, this);

    control_enable_sub_.subscribe(
        "enable", 1, nullptr,
        []() { ROS_WARN_STREAM("control enable message timeouted"); },
        ros::Duration{1.0});
    control_enable_sub_.set_default_message(std_msgs::Bool{});

    wrench_pub_ = nh_.advertise<geometry_msgs::Wrench>("wrench", 1);
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

      const auto control_output =
          controller_->control(state_, desired_state_, d_state_, dt);

      const auto wrench_msg =
          is_control_enabled()
              ? auv::common::conversions::convert<ControllerBase::WrenchVector,
                                                  geometry_msgs::Wrench>(
                    control_output)
              : geometry_msgs::Wrench{};

      wrench_pub_.publish(wrench_msg);
    }
  }

 private:
  bool is_control_enabled() { return control_enable_sub_.get_message().data; }

  void odometry_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    state_ =
        auv::common::conversions::convert<nav_msgs::Odometry,
                                          ControllerBase::StateVector>(*msg);
    d_state_.head(6) = state_.tail(6);
  }

  void cmd_vel_callback(const geometry_msgs::Twist::ConstPtr& msg) {
    desired_state_.tail(6) =
        auv::common::conversions::convert<geometry_msgs::Twist,
                                          ControllerBase::Vector>(*msg);
  }

  void cmd_pose_callback(const geometry_msgs::Pose::ConstPtr& msg) {
    desired_state_.head(6) =
        auv::common::conversions::convert<geometry_msgs::Pose,
                                          ControllerBase::Vector>(*msg);
  }

  void imu_callback(const sensor_msgs::Imu::ConstPtr& msg) {
    d_state_(6) = msg->linear_acceleration.x;
    d_state_(7) = msg->linear_acceleration.y;
    d_state_(8) = msg->linear_acceleration.z;
    d_state_.tail(3) = Eigen::Vector3d::Zero();
  }

  ros::Rate rate_;
  ros::NodeHandle nh_;
  ros::Subscriber odometry_sub_;
  ros::Subscriber cmd_vel_sub_;
  ros::Subscriber cmd_pose_sub_;
  ros::Subscriber imu_sub_;
  ros::Publisher wrench_pub_;
  ControlEnableSub control_enable_sub_;
  ControllerBasePtr controller_;

  ControllerBase::StateVector state_{ControllerBase::StateVector::Zero()};
  ControllerBase::StateVector desired_state_{
      ControllerBase::StateVector::Zero()};
  ControllerBase::StateVector d_state_{ControllerBase::StateVector::Zero()};

  ControllerLoader controller_loader_{"auv_controllers",
                                      "auv::control::SixDOFControllerBase"};
};

}  // namespace control
}  // namespace auv
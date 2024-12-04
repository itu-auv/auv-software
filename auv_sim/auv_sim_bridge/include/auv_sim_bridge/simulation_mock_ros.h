#ifndef _AUV_SIM_BRIDGE_SIMULATION_MOCK_H_
#define _AUV_SIM_BRIDGE_SIMULATION_MOCK_H_
#include <boost/array.hpp>

#include "auv_msgs/MotorCommand.h"
#include "auv_msgs/Power.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/TwistWithCovarianceStamped.h"
#include "sensor_msgs/FluidPressure.h"
#include "sim_thruster_ros.h"
#include "std_msgs/Float32.h"
#include "std_srvs/SetBool.h"
#include "uuv_gazebo_ros_plugins_msgs/FloatStamped.h"
#include "uuv_sensor_ros_plugins_msgs/DVL.h"
#include "uuv_sensor_ros_plugins_msgs/DVLBeam.h"

namespace auv_sim_bridge {

class SimulationMockROS {
  const size_t kThrusterSize = 8;

 private:
  ros::NodeHandle nh_;
  ros::Rate rate_;  /// simulation control rate

  ros::Subscriber direct_drive_sub_;
  ros::Subscriber depth_sub_;
  ros::Subscriber dvl_sub_;
  ros::Publisher depth_pub_;
  ros::Publisher altitude_pub_;
  ros::Publisher velocity_pub_;
  ros::Publisher velocity_stamped_pub_;
  ros::Publisher battery_sim_pub_;
  ros::ServiceServer arm_srv_;
  ros::ServiceServer set_dvl_ping_srv_;

  std::vector<SimThruster> thrusters_;  /// thruster UUV interface
  bool armed_;

  double gravity_;            /// m/s^2
  double density_;            /// kg / m^3
  double standard_pressure_;  /// kPa
  double battery_voltage_;    /// V
  double battery_current_;    /// A

  bool setArmHandler(std_srvs::SetBoolRequest &req,
                     std_srvs::SetBoolResponse &resp) {
    armed_ = req.data;
    resp.success = true;
    ROS_INFO("AUV armed state set to: %s", armed_ ? "true" : "false");
    return true;
  }

  bool setDVLPing(std_srvs::SetBoolRequest &req,
                  std_srvs::SetBoolResponse &resp) {
    // this is a dummy service, does nothing
    resp.success = true;
    ROS_INFO("DVL ping set request received.");
    return true;
  }

  void directDriveCallback(const auv_msgs::MotorCommand &msg) {
    const auto stamp = ros::Time::now();
    for (int i = 0; i < kThrusterSize; i++) {
      if (i < msg.channels.size()) {
        if (armed_) {
          thrusters_[i].publish(msg.channels.at(i), stamp);
        } else {
          thrusters_[i].publish(1500, stamp);
        }
      } else {
        ROS_WARN("Received MotorCommand with insufficient channels.");
      }
    }
  }

  void depthCallback(const sensor_msgs::FluidPressure &msg) {
    const double fluid_pressure = msg.fluid_pressure;
    const double depth =
        1000.0 * (fluid_pressure - standard_pressure_) / (gravity_ * density_);
    std_msgs::Float32 depth_msg;
    depth_msg.data = -depth;
    depth_pub_.publish(depth_msg);
  }

  void dvlCallback(const uuv_sensor_ros_plugins_msgs::DVL &msg) {
    std_msgs::Float32 altitude_msg;
    geometry_msgs::Twist velocity_msg;
    geometry_msgs::TwistWithCovarianceStamped velocity_stamped_msg;
    // Dvl altitude
    altitude_msg.data = msg.altitude;
    altitude_pub_.publish(altitude_msg);

    // Dvl velocity
    velocity_msg.linear = msg.velocity;
    velocity_pub_.publish(velocity_msg);

    boost::array<double, 36> covariance;
    covariance.fill(1e-6);
    velocity_stamped_msg.header.frame_id = "dvl_link";
    velocity_stamped_msg.twist.twist = velocity_msg;
    velocity_stamped_msg.twist.covariance = covariance;
    velocity_stamped_pub_.publish(velocity_stamped_msg);
  }

  void initializeParameters() {
    ros::NodeHandle nh_priv("~");
    double rate;
    if (nh_priv.getParam("rate", rate)) {
      rate_ = ros::Rate(rate);
    } else {
      rate_ = ros::Rate(20.0);
      ROS_WARN("Parameter 'rate' not set. Using default: 20.0");
    }

    if (!nh_.getParam("/env/standard_pressure", standard_pressure_)) {
      standard_pressure_ = 101325.0 / 1000.0;  // convert to kPa
      ROS_WARN(
          "Parameter '/env/standard_pressure' not set. Using default: 101.325 "
          "kPa");
    } else {
      standard_pressure_ /= 1000.0;  // convert to kPa
    }

    if (!nh_.getParam("/env/gravity", gravity_)) {
      gravity_ = 9.81;
      ROS_WARN("Parameter '/env/gravity' not set. Using default: 9.81 m/s^2");
    }

    if (!nh_.getParam("/env/density", density_)) {
      density_ = 1000.0;
      ROS_WARN(
          "Parameter '/env/density' not set. Using default: 1000.0 kg/m^3");
    }

    if (!nh_priv.getParam("battery_voltage", battery_voltage_)) {
      battery_voltage_ = 14;
      ROS_WARN("Parameter 'battery_voltage' not set. Using default: 14 V");
    }

    if (!nh_priv.getParam("battery_current", battery_current_)) {
      battery_current_ = 5;
      ROS_WARN("Parameter 'battery_current' not set. Using default: 5 A");
    }
  }

  void initializePublishers() {
    depth_pub_ = nh_.advertise<std_msgs::Float32>("depth", 1);
    altitude_pub_ = nh_.advertise<std_msgs::Float32>("altitude", 1);
    velocity_pub_ = nh_.advertise<geometry_msgs::Twist>("velocity", 1);
    velocity_stamped_pub_ =
        nh_.advertise<geometry_msgs::TwistWithCovarianceStamped>(
            "velocity_stamped", 1);
    battery_sim_pub_ = nh_.advertise<auv_msgs::Power>("power", 1);
  }

  void initializeSubscribers() {
    depth_sub_ =
        nh_.subscribe("pressure", 1, &SimulationMockROS::depthCallback, this);
    dvl_sub_ = nh_.subscribe("dvl", 1, &SimulationMockROS::dvlCallback, this);
    direct_drive_sub_ = nh_.subscribe(
        "direct_drive", 1, &SimulationMockROS::directDriveCallback, this);
  }

  void initializeServices() {
    arm_srv_ = nh_.advertiseService("set_arming",
                                    &SimulationMockROS::setArmHandler, this);
    set_dvl_ping_srv_ = nh_.advertiseService(
        "sensors/dvl/set_ping", &SimulationMockROS::setDVLPing, this);
  }

  void initializeThrusters() {
    thrusters_.reserve(kThrusterSize);
    for (int i = 0; i < kThrusterSize; ++i) {
      thrusters_.emplace_back(nh_, i);
    }
  }

 public:
  SimulationMockROS(const ros::NodeHandle &nh)
      : nh_(nh), rate_(1.0), armed_(false) {
    initializeParameters();
    initializePublishers();
    initializeSubscribers();
    initializeServices();
    initializeThrusters();
  }

  void spin() {
    // cumulative current draw sum up all thrusters
    float current_draw = 0.0f;
    std::for_each(thrusters_.begin(), thrusters_.end(),
                  [&current_draw](const SimThruster &thruster) {
                    current_draw += thruster.get_current_draw();
                  });

    auv_msgs::Power battery_msg;
    battery_msg.voltage = battery_voltage_;
    battery_msg.current = current_draw;
    battery_msg.power = std::fabs(battery_voltage_ * battery_current_);
    while (ros::ok()) {
      battery_sim_pub_.publish(battery_msg);

      ros::spinOnce();
      rate_.sleep();
    }
  }
};

};  // namespace auv_sim_bridge

#endif

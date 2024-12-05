#ifndef _AUV_SIM_BRIDGE_SIMULATION_MOCK_H_
#define _AUV_SIM_BRIDGE_SIMULATION_MOCK_H_
#include <boost/array.hpp>

#include "auv_msgs/MotorCommand.h"
#include "auv_msgs/Power.h"
#include "geometry_msgs/Twist.h"
#include "sensor_msgs/FluidPressure.h"
#include "sim_thruster_ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include "std_srvs/SetBool.h"
#include "uuv_sensor_ros_plugins_msgs/DVL.h"

namespace auv_sim_bridge {

class SimulationMockROS {
  const size_t kThrusterSize = 8;

 private:
  ros::NodeHandle nh_;
  ros::Rate rate_;  /// simulation control rate

  ros::Subscriber drive_pulse_sub_;
  ros::Subscriber depth_sub_;
  ros::Subscriber dvl_sub_;
  ros::Publisher depth_pub_;
  ros::Publisher altitude_pub_;
  ros::Publisher velocity_raw_pub_;
  ros::Publisher is_valid_pub_;
  ros::Publisher battery_sim_pub_;
  ros::ServiceServer dvl_enable_srv_;

  std::vector<SimThruster> thrusters_;  /// thruster UUV interface

  double gravity_;            /// m/s^2
  double density_;            /// kg / m^3
  double standard_pressure_;  /// kPa
  double battery_voltage_;    /// V
  double battery_current_;    /// A
  bool dvl_enabled_;

  bool setDVLEnable(std_srvs::SetBoolRequest &req,
                    std_srvs::SetBoolResponse &resp) {
    dvl_enabled_ = req.data;
    resp.success = true;
    ROS_INFO("DVL enable set request received. DVL enabled: %s",
             dvl_enabled_ ? "true" : "false");
    return true;
  }

  void drivePulseCallback(const auv_msgs::MotorCommand &msg) {
    const auto stamp = ros::Time::now();

    // Remapping array: maps input channels to specific thruster indices
    std::vector<int> channelRemap = {1, 7, 2, 5, 0, 6, 3, 4};

    if (msg.channels.size() < kThrusterSize) {
      ROS_WARN("Received MotorCommand with insufficient channels.");
      return;
    }

    for (int i = 0; i < kThrusterSize; i++) {
      if (i < channelRemap.size() && channelRemap[i] < msg.channels.size()) {
        int remappedChannel = channelRemap[i];
        thrusters_[i].publish(msg.channels.at(remappedChannel), stamp);
      } else {
        ROS_WARN(
            "Invalid channel remapping or insufficient channels for "
            "remapping.");
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
    geometry_msgs::Twist velocity_raw_msg;

    // Dvl altitude
    altitude_msg.data = msg.altitude;
    altitude_pub_.publish(altitude_msg);

    // Dvl velocity
    velocity_raw_msg.linear = msg.velocity;
    velocity_raw_pub_.publish(velocity_raw_msg);

    // Dvl validity
    std_msgs::Bool is_valid_msg;
    is_valid_msg.data = dvl_enabled_;
    is_valid_pub_.publish(is_valid_msg);
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
      battery_voltage_ = 16;
      ROS_WARN("Parameter 'battery_voltage' not set. Using default: 16 V");
    }

    if (!nh_priv.getParam("battery_current", battery_current_)) {
      battery_current_ = 10;
      ROS_WARN("Parameter 'battery_current' not set. Using default: 10 A");
    }

    dvl_enabled_ = true;
  }

  void initializePublishers() {
    depth_pub_ = nh_.advertise<std_msgs::Float32>("depth", 1);
    altitude_pub_ = nh_.advertise<std_msgs::Float32>("altitude", 1);
    velocity_raw_pub_ = nh_.advertise<geometry_msgs::Twist>("velocity_raw", 1);
    is_valid_pub_ = nh_.advertise<std_msgs::Bool>("is_valid", 1);
    battery_sim_pub_ = nh_.advertise<auv_msgs::Power>("power", 1);
  }

  void initializeSubscribers() {
    depth_sub_ =
        nh_.subscribe("pressure", 1, &SimulationMockROS::depthCallback, this);
    dvl_sub_ = nh_.subscribe("dvl", 1, &SimulationMockROS::dvlCallback, this);
    drive_pulse_sub_ = nh_.subscribe(
        "drive_pulse", 1, &SimulationMockROS::drivePulseCallback, this);
  }

  void initializeServices() {
    dvl_enable_srv_ = nh_.advertiseService(
        "sensors/dvl/enable", &SimulationMockROS::setDVLEnable, this);
  }

  void initializeThrusters() {
    thrusters_.reserve(kThrusterSize);
    for (int i = 0; i < kThrusterSize; ++i) {
      thrusters_.emplace_back(nh_, i);
    }
  }

 public:
  SimulationMockROS(const ros::NodeHandle &nh) : nh_(nh), rate_(1.0) {
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

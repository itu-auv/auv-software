#ifndef AUV_SIM_BRIDGE__SIMULATION_MOCK_ROS_H_
#define AUV_SIM_BRIDGE__SIMULATION_MOCK_ROS_H_
#include <Eigen/Dense>
#include <boost/array.hpp>

#include "geometry_msgs/Twist.h"
#include "nav_msgs/Odometry.h"
#include "sensor_msgs/BatteryState.h"
#include "sensor_msgs/FluidPressure.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Range.h"
#include "sim_thruster_ros.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float32.h"
#include "std_msgs/UInt16MultiArray.h"
#include "std_srvs/SetBool.h"
#include "uuv_sensor_ros_plugins_msgs/DVL.h"

namespace auv_sim_bridge {

class SimulationMockROS {
  const size_t kThrusterSize = 8;
  const double kMinDvlAltitude = 0.3;

 private:
  ros::NodeHandle nh_;
  ros::Rate rate_;  /// simulation control rate

  ros::Subscriber drive_pulse_sub_;
  ros::Subscriber depth_sub_;
  ros::Subscriber dvl_sub_;
  ros::Subscriber imu_sub_;
  ros::Publisher depth_pub_;
  ros::Publisher altitude_pub_;
  ros::Publisher velocity_raw_pub_;
  ros::Publisher is_valid_pub_;
  ros::Publisher imu_pub_;
  ros::Publisher battery_sim_pub_;
  ros::ServiceServer dvl_enable_srv_;

  std::vector<SimThruster> thrusters_;  /// thruster UUV interface
  std::vector<double> imu_offset_;

  double gravity_;            /// m/s^2
  double density_;            /// kg / m^3
  double standard_pressure_;  /// kPa
  double battery_voltage_;    /// V
  double battery_current_;    /// A
  bool dvl_enabled_;
  Eigen::Matrix3d linear_covariance_;
  Eigen::Matrix3d noise_transform_;

  void rotateVelocity(geometry_msgs::Twist &input_velocity, double angle) {
    const double theta = angle * M_PI / 180.0;
    Eigen::Matrix3d rotation_matrix;
    rotation_matrix << std::cos(theta), -std::sin(theta), 0, std::sin(theta),
        std::cos(theta), 0, 0, 0, 1;

    Eigen::Vector3d velocity_vector(input_velocity.linear.x,
                                    input_velocity.linear.y,
                                    input_velocity.linear.z);

    Eigen::Vector3d rotated_velocity = rotation_matrix * velocity_vector;

    input_velocity.linear.x = rotated_velocity.x();
    input_velocity.linear.y = rotated_velocity.y();
    input_velocity.linear.z = rotated_velocity.z();
  }

  void addNoiseToTwist(geometry_msgs::Twist &input) {
    Eigen::Vector3d noise = noise_transform_ * Eigen::Vector3d::Random();

    input.linear.x += noise(0);
    input.linear.y += noise(1);
    input.linear.z += noise(2);
  }

  bool setDVLEnable(std_srvs::SetBoolRequest &req,
                    std_srvs::SetBoolResponse &resp) {
    dvl_enabled_ = req.data;
    resp.success = true;
    ROS_INFO("DVL enable set request received. DVL enabled: %s",
             dvl_enabled_ ? "true" : "false");
    return true;
  }

  void drivePulseCallback(const std_msgs::UInt16MultiArray &msg) {
    const auto stamp = ros::Time::now();

    if (msg.data.size() < kThrusterSize) {
      ROS_WARN("Received UInt16MultiArray with insufficient data.");
      return;
    }

    for (int i = 0; i < kThrusterSize; i++) {
      thrusters_[i].publish(msg.data.at(i), stamp);
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

  void imuCallback(const sensor_msgs::Imu &msg) {
    sensor_msgs::Imu imu_msg = msg;

    imu_msg.angular_velocity.x += imu_offset_[0];
    imu_msg.angular_velocity.y += imu_offset_[1];
    imu_msg.angular_velocity.z += imu_offset_[2];

    imu_pub_.publish(imu_msg);
  }

  void dvlCallback(const uuv_sensor_ros_plugins_msgs::DVL &msg) {
    geometry_msgs::Twist velocity_raw_msg;
    std_msgs::Bool is_valid_msg;
    std_msgs::Float32 altitude_msg;

    if (msg.altitude > kMinDvlAltitude && dvl_enabled_) {
      velocity_raw_msg.linear = msg.velocity;
      addNoiseToTwist(velocity_raw_msg);
      rotateVelocity(velocity_raw_msg, 45.0);

      is_valid_msg.data = true;
    }
    altitude_msg.data = msg.altitude;

    altitude_pub_.publish(altitude_msg);
    velocity_raw_pub_.publish(velocity_raw_msg);
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

    if (!nh_priv.getParam("imu/drift", imu_offset_)) {
      imu_offset_ = {0.0, 0.0, 0.0};
      ROS_WARN(
          "Parameter 'imu_offset_' not set. Using default: {0.0, 0.0, "
          "0.0}");
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

    // DVL covariance
    if (!nh_.getParam("sensors/dvl/covariance/linear_x",
                      linear_covariance_(0, 0))) {
      linear_covariance_(0, 0) = 0.000015;
      ROS_WARN(
          "Parameter 'sensors/dvl/covariance/linear_x' not set. Using default: "
          "0.000015");
    }
    if (!nh_.getParam("sensors/dvl/covariance/linear_y",
                      linear_covariance_(1, 1))) {
      linear_covariance_(1, 1) = 0.000015;
      ROS_WARN(
          "Parameter 'sensors/dvl/covariance/linear_y' not set. Using default: "
          "0.000015");
    }
    if (!nh_.getParam("sensors/dvl/covariance/linear_z",
                      linear_covariance_(2, 2))) {
      linear_covariance_(2, 2) = 0.00005;
      ROS_WARN(
          "Parameter 'sensors/dvl/covariance/linear_z' not set. Using default: "
          "0.00005");
    }

    // Compute eigendecomposition once
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(
        linear_covariance_);
    if (eigen_solver.info() != Eigen::Success) {
      ROS_WARN("Failed to compute eigenvalues for the covariance matrix!");
      noise_transform_ = Eigen::Matrix3d::Zero();
    } else {
      noise_transform_ =
          eigen_solver.eigenvectors() *
          eigen_solver.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
    }
  }

  void initializePublishers() {
    depth_pub_ = nh_.advertise<std_msgs::Float32>("depth", 1);
    imu_pub_ = nh_.advertise<sensor_msgs::Imu>("imu_drifted", 1);
    altitude_pub_ = nh_.advertise<std_msgs::Float32>("altitude", 1);
    velocity_raw_pub_ = nh_.advertise<geometry_msgs::Twist>("velocity_raw", 1);
    is_valid_pub_ = nh_.advertise<std_msgs::Bool>("is_valid", 1);
    battery_sim_pub_ = nh_.advertise<sensor_msgs::BatteryState>("power", 1);
  }

  void initializeSubscribers() {
    depth_sub_ =
        nh_.subscribe("pressure", 1, &SimulationMockROS::depthCallback, this);
    imu_sub_ =
        nh_.subscribe("imu_raw", 1, &SimulationMockROS::imuCallback, this);
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

    sensor_msgs::BatteryState battery_msg;
    battery_msg.voltage = battery_voltage_;
    battery_msg.current = current_draw;
    battery_msg.power_supply_status =
        sensor_msgs::BatteryState::POWER_SUPPLY_STATUS_DISCHARGING;
    battery_msg.present = true;
    while (ros::ok()) {
      battery_sim_pub_.publish(battery_msg);

      ros::spinOnce();
      rate_.sleep();
    }
  }
};

};  // namespace auv_sim_bridge

#endif  // AUV_SIM_BRIDGE__SIMULATION_MOCK_ROS_H_

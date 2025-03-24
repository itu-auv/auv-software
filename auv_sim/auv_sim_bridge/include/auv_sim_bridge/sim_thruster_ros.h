#pragma once

#include "auv_msgs/MotorCommand.h"
#include "ros/ros.h"
#include "uuv_gazebo_ros_plugins_msgs/FloatStamped.h"

class SimThruster {
 private:
  size_t id_;
  ros::NodeHandle nh_;
  ros::Publisher thruster_pub_;
  ros::Subscriber thrust_sub_;
  float current_draw_{0.0f};

 public:
  using Ptr = std::shared_ptr<SimThruster>;

  void publish(const double thrust, const ros::Time &stamp) {
    uuv_gazebo_ros_plugins_msgs::FloatStamped msg;
    msg.data = thrust;
    msg.header.stamp = stamp;
    publish(msg);
  }

  void publish(const double thrust) { publish(thrust, ros::Time::now()); }

  void publish(const uuv_gazebo_ros_plugins_msgs::FloatStamped &msg) {
    thruster_pub_.publish(msg);
  }

  void thrust_callback(
      const uuv_gazebo_ros_plugins_msgs::FloatStamped::ConstPtr &msg) {
    current_draw_ = msg->data / 10.0;  // TODO: add realistic current draw
  }

  float get_current_draw() const { return current_draw_; }

  SimThruster(const ros::NodeHandle &nh, const size_t id) : nh_(nh), id_(id) {
    const auto topic =
        std::string{"thrusters/"} + std::to_string(id_) + std::string{"/input"};
    thruster_pub_ =
        nh_.advertise<uuv_gazebo_ros_plugins_msgs::FloatStamped>(topic, 1);

    nh_.subscribe("thrusters/" + std::to_string(id_) + "/thrust", 1,
                  &SimThruster::thrust_callback, this);
  }
};

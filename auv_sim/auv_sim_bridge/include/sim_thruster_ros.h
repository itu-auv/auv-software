#include "auv_msgs/MotorCommand.h"
#include "ros/ros.h"
#include "uuv_gazebo_ros_plugins_msgs/FloatStamped.h"

class SimThruster {
 private:
  size_t id_;
  ros::NodeHandle nh_;
  ros::Publisher thruster_pub_;

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

  SimThruster(const ros::NodeHandle &nh, const size_t id) : nh_(nh), id_(id) {
    // /turquoise/thrusters/'id'/input
    std::string topic = "thrusters/" + std::to_string(id_) + "/input";
    thruster_pub_ =
        nh_.advertise<uuv_gazebo_ros_plugins_msgs::FloatStamped>(topic, 1);
  }
};
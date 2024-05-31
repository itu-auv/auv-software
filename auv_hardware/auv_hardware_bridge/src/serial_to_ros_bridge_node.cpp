#include "auv_hardware_bridge/serial_to_ros_bridge.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "serial_to_ros_bridge");
  ros::NodeHandle nh;

  auv_hardware::SerialToROSBridge bridge(nh);
  bridge.spin();

  return 0;
}

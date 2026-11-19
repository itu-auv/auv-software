#include "auv_control/thruster_manager_ros.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto nh = std::make_shared<rclcpp::Node>("thruster_manager_node");
  auv::control::ThrusterManagerROS manager(nh);
  manager.spin();

  if (rclcpp::ok()) rclcpp::shutdown();
  return 0;
}

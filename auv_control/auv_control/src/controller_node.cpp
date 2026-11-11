#include "auv_control/controller_ros.h"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto nh = std::make_shared<rclcpp::Node>("auv_control_node");
  auv::control::ControllerROS controller(nh);

  controller.spin();

  rclcpp::shutdown();
  return 0;
}

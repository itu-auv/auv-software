#include "auv_canbus_bridge/auv_canbus_bridge_ros.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "auv_canbus_bridge_node");
  auto node_handle = ros::NodeHandle{};

  try {
    auto canbus_ros_bridge = auv_hardware::CanbusBridgeROS{node_handle};
    canbus_ros_bridge.spin();
  } catch (const std::exception &e) {
    ROS_FATAL("Exception: %s", e.what());
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

#include <thread>

#include "auv_mapping/object_map_tf_server_ros.hpp"

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node_handle = std::make_shared<rclcpp::Node>("object_map_tf_server");
  auto server = auv_mapping::ObjectMapTFServerROS{node_handle};

  auto executor =
      rclcpp::executors::MultiThreadedExecutor{rclcpp::ExecutorOptions{}, 2};
  executor.add_node(node_handle);
  auto spinner = std::thread{[&executor]() { executor.spin(); }};

  server.run();
  rclcpp::shutdown();
  spinner.join();

  return 0;
}

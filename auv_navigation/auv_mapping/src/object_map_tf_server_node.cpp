#include "auv_mapping/object_map_tf_server_ros.hpp"

int main(int argc, char **argv) {
  ros::init(argc, argv, "object_map_tf_server");
  auto node_handle = ros::NodeHandle{};
  auto server = auv_mapping::ObjectMapTFServerROS{node_handle};

  auto spinner = ros::AsyncSpinner{2};
  spinner.start();

  server.run();
  ros::waitForShutdown();

  return 0;
}

#include "auv_control/thruster_manager_ros.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "thruster_manager_node");

  ros::NodeHandle nh;
  auv::control::ThrusterManagerROS manager(nh);
  manager.spin();

  return 0;
}

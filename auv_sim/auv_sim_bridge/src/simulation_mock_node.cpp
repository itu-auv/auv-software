#include "../include/simulation_mock_ros.h"
#include "ros/ros.h"

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "simulation_mock_node");
  ros::NodeHandle nh;
  auv_sim_bridge::SimulationMockROS node(nh);
  node.spin();
  return 0;
}

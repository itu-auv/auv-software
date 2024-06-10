#include "auv_control/controller_ros.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "controller_node");

  ros::NodeHandle nh;
  auv::control::ControllerROS controller(nh);

  controller.spin();

  return 0;
}

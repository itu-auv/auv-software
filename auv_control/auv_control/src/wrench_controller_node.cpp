#include <ros/ros.h>

#include "auv_control/wrench_controller.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "wrench_controller");
  ros::NodeHandle nh;
  auv::control::WrenchController controller(nh);
  controller.spin();
  return 0;
}

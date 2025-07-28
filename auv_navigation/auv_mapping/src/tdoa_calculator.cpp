#include <auv_mapping/tdoa_calculator.hpp>

int main(int argc, char** argv) {
  ros::init(argc, argv, "tdoa_localizer_node");
  ros::NodeHandle nh("~");
  auto node = TDOALocalizer(nh);
  ros::spin();
  return 0;
}

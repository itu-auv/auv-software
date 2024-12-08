#include <array>

#include "auv_canbus_bridge/canbus/helper.hpp"
#include "auv_canbus_bridge/canbus_socket.hpp"
#include "auv_canbus_bridge/modules.hpp"

class CANBridgeTest {
 public:
  using ModuleBase = auv_hardware::canbus::modules::ModuleBase;
  using DrivePulseModule = auv_hardware::canbus::modules::DrivePulseModule;

  explicit CANBridgeTest(const ros::NodeHandle &node_handle)
      : node_handle_{node_handle},
        socket_{},
        interface_name_{""},
        drive_pulse_module_{node_handle_, socket_} {
    auto node_handle_private = ros::NodeHandle{"~"};
    node_handle_private.param<std::string>("interface", interface_name_,
                                           "vcan0");

    socket_.initialize(interface_name_);
  }

 private:
  ros::NodeHandle node_handle_;
  auv_hardware::CanbusSocket socket_;
  std::string interface_name_;
  DrivePulseModule drive_pulse_module_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "can_bridge_test");
  ros::NodeHandle nh;
  try {
    CANBridgeTest can_bridge(nh);
    ros::spin();
  } catch (const std::exception &e) {
    ROS_FATAL("Exception: %s", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

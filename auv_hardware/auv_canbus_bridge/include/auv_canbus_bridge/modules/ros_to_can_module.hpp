#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/QuadDrivePulse.h"
#include "auv_msgs/MotorCommand.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class DVLModule : public ModuleBase {
 public:
  DVLModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    ROS_INFO_STREAM("Initialized DVL Module for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override{
      //
  };

 private:
  void dvl_rosmsg_callback(const auv_msgs::MotorCommand::ConstPtr &msg) {
    //
  }

  ros::Subscriber serial_ros_subscriber;
  ros::Publisher canbus_publisher;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

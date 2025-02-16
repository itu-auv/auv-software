#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/LatchedServo.h"
#include "ros/ros.h"
#include "std_srvs/Trigger.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class MarkerDropperModule : public ModuleBase {
  constexpr static auto kSetServoExtendedId =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::Mainboard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::LatchedServoCommand);

 public:
  MarkerDropperModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    marker_dropper_service = ModuleBase::node_handle().advertiseService(
        "actuators/ball_dropper/drop",
        &MarkerDropperModule::marker_dropper_handler, this);

    ROS_INFO_STREAM("Initialized MarkerDropperModule for CANBUS");
  }

  bool marker_dropper_handler(std_srvs::Trigger::Request &req,
                              std_srvs::Trigger::Response &res) {
    res.success = drop_marker(kSetServoExtendedId);
    res.message = res.success ? "Marker dropped" : "Failed to drop marker";
    return true;
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override{
      //
  };

 protected:

  bool drop_marker(auv_hardware::canbus::Identifier id, const int channel = 3) {
    bool success = true;

    auto set_servo = auv_canbus_msgs::LatchedServo();
    set_servo.pulse = 2300;
    set_servo.channel = channel;
    success = dispatch_message(id, set_servo);

    ros::Duration(2.0).sleep();

    auto reset_servo = auv_canbus_msgs::LatchedServo();
    reset_servo.pulse = 700;
    reset_servo.channel = channel;
    success &= dispatch_message(id, reset_servo);

    return success;
  }

 private:
  ros::ServiceServer marker_dropper_service;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/QuadDrivePulse.h"
#include "auv_msgs/MotorCommand.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class DrivePulseModule : public ModuleBase {
 public:
  constexpr static auto kFirstQuadDriveExtendedId =
      auv_hardware::canbus::make_extended_id(
          0x01, auv_hardware::canbus::NodeID::PropulsionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::FirstQuadThrusterCommand);

  constexpr static auto kSecondQuadDriveExtendedId =
      auv_hardware::canbus::make_extended_id(
          0x01, auv_hardware::canbus::NodeID::PropulsionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::SecondQuadThrusterCommand);

  DrivePulseModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    drive_subscriber_ = ModuleBase::node_handle().subscribe(
        "board/drive_pulse", 10, &DrivePulseModule::drive_pulse_callback, this);

    ROS_INFO_STREAM("Initialized DrivePulseModule for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override {
    //
  };

 private:
  void drive_pulse_callback(const auv_msgs::MotorCommand::ConstPtr &msg) {
    auv_canbus_msgs::QuadDrivePulse pulse_msg;

    std::copy(msg->channels.begin(), msg->channels.begin() + 4,
              pulse_msg.pulse_width.begin());

    dispatch_message(kFirstQuadDriveExtendedId, pulse_msg);

    std::copy(msg->channels.begin() + 4, msg->channels.end(),
              pulse_msg.pulse_width.begin());

    dispatch_message(kSecondQuadDriveExtendedId, pulse_msg);
  }

  ros::Subscriber drive_subscriber_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

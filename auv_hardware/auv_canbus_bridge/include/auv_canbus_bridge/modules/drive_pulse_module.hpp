#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
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

  virtual void on_received_message(auv_hardware::canbus::ExtendedIdType id,
                                   const std::vector<uint8_t> &data) override {
    //
  };

 private:
  void drive_pulse_callback(const auv_msgs::MotorCommand::ConstPtr &msg) {
    std::vector<uint8_t> data;
    data.resize(8);

    auto pack_pulse = [&data, &msg](const int offset) {
      for (int i = 0; i < 4; ++i) {
        const auto pulse = msg->channels[offset + i];
        data[2 * i] = pulse & 0xFF;
        data[2 * i + 1] = (pulse >> 8) & 0xFF;
      }
    };

    pack_pulse(0);

    dispatch_message(kFirstQuadDriveExtendedId, data);

    pack_pulse(4);

    dispatch_message(kSecondQuadDriveExtendedId, data);
  }

  ros::Subscriber drive_subscriber_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

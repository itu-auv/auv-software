#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "ros/ros.h"
#include "std_msgs/Bool.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class KillswitchReportModule : public ModuleBase {
  static constexpr auto kPropulsionBoardStatusReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::PropulsionBoardStatusReport);

 public:
  KillswitchReportModule(const ros::NodeHandle &node_handle,
                         CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    status_report_publisher_ =
        ModuleBase::node_handle().advertise<std_msgs::Bool>(
            "propulsion_board/status", 10);
    ROS_INFO_STREAM("Initialized KillswitchReportModule for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override {
    if (id != kPropulsionBoardStatusReportIdentifier) {
      return;
    }

    if (data.size() != 1) {
      ROS_WARN_STREAM("Received invalid status report message");
      return;
    }

    auto status_msg = std_msgs::Bool{};
    status_msg.data = data[0];

    status_report_publisher_.publish(status_msg);
  };

 private:
  ros::Publisher status_report_publisher_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

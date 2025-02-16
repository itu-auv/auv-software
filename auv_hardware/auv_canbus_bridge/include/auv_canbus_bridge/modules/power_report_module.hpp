#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/Power.h"
#include "auv_msgs/Power.h"
#include "ros/ros.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class PowerReportModule : public ModuleBase {
  static constexpr auto kPowerReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::MainboardPowerReport);

 public:
  PowerReportModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    power_report_publisher_ =
        ModuleBase::node_handle().advertise<auv_msgs::Power>(
            "mainboard/power_sensor/power", 10);
    ROS_INFO_STREAM("Initialized PowerReportModule for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override {
    if (id != kPowerReportIdentifier) {
      return;
    }

    if (data.size() != 8) {
      ROS_WARN_STREAM("Received invalid power report message");
      return;
    }

    auto report = auv_canbus_msgs::Power{};
    std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&report));

    auto power_msg = auv_msgs::Power{};

    power_msg.voltage = report.voltage;
    power_msg.current = report.current;
    power_msg.power = std::abs(report.voltage * report.current);

    power_report_publisher_.publish(power_msg);
  };

 private:
  ros::Publisher power_report_publisher_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

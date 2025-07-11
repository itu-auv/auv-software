#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/Pressure.h"
#include "canbus/f16_converter.hpp"
#include "ros/ros.h"
#include "std_msgs/Float32.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class PressureReportModule : public ModuleBase {
  static constexpr auto kPressureReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::PressureSensorReport);

 public:
  PressureReportModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    power_report_publisher_ =
        ModuleBase::node_handle().advertise<auv_msgs::Power>(
            "mainboard/pressure_sensor/data", 10);
    ROS_INFO_STREAM("Initialized PressureReportModule for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override {
    if (id != kPressureReportIdentifier) {
      return;
    }

    if (data.size() != 8) {
      ROS_WARN_STREAM("Received invalid pressure report message");
      return;
    }

    auto report = auv_canbus_msgs::Pressure{};
    std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&report));

    const auto depth_msg = std_msgs::Float32{auv::math::from_f16(report.depth)};
    depth_publisher.publish(depth_msg);

    const auto external_pressure_msg =
        std_msgs::Float32{auv::math::from_f16(report.pressure)};
    external_pressure_publisher.publish(external_pressure_msg);

    const auto external_temperature_msg =
        std_msgs::Float32{auv::math::from_f16(report.temperature)};
    external_temperature_publisher.publish(external_temperature_msg);
  };

 private:
  ros::Publisher depth_publisher;
  ros::Publisher external_pressure_publisher;
  ros::Publisher external_temperature_publisher
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

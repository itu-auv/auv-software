#pragma once
#include "auv_canbus_bridge/modules/module_base.hpp"
#include "auv_canbus_msgs/DistanceMeasurement.h"
#include "auv_canbus_msgs/SonarActivation.h"
#include "ros/ros.h"
#include "sensor_msgs/Range.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class PingSonarModule : public ModuleBase {
  static constexpr auto KRightSonarReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::RightSonarReport);

  static constexpr auto kFrontSonarReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::FrontSonarReport);

  static constexpr auto kBackReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::BackSonarReport);

  static constexpr auto kLeftReportIdentifier =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::ExpansionBoard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::LeftSonarReport);

  static constexpr auto kSonarActivationCommand =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::Mainboard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::SonarActivationCommand);

 public:
  PingSonarModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    front_sonar_publisher =
        ModuleBase::node_handle().advertise<sensor_msgs::Range>(
            "sensors/sonar_front/range", 10);
    back_sonar_publisher =
        ModuleBase::node_handle().advertise<sensor_msgs::Range>(
            "sensors/sonar_back/range", 10);
    left_sonar_publisher =
        ModuleBase::node_handle().advertise<sensor_msgs::Range>(
            "sensors/sonar_left/range", 10);
    right_sonar_publisher =
        ModuleBase::node_handle().advertise<sensor_msgs::Range>(
            "sensors/sonar_right/range", 10);
    ModuleBase::node_handle().advertiseService(
        "acoustics/sonars/enable_disable", &PingSonarModule::activate_sonar, this);

    ROS_INFO_STREAM("Initialized PingSonarModule for CANBUS");
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override {
    if (!is_sonar_id(id) || !is_valid_data(data)) {
      return;
    }

    // TODO Add deserialization
    auto sonar_data = auv_canbus_msgs::DistanceMeasurement{};
    std::copy(data.begin(), data.end(),
              reinterpret_cast<uint8_t *>(&sonar_data));

    if (id == kFrontSonarReportIdentifier) {
      publish_sonar_data(front_sonar_publisher,
                         "taluy/base_link/sonar_front_link",
                         sonar_data.distance);
    }
    if (id == kBackReportIdentifier) {
      publish_sonar_data(back_sonar_publisher,
                         "taluy/base_link/sonar_back_link",
                         sonar_data.distance);
    }
    if (id == kLeftReportIdentifier) {
      publish_sonar_data(left_sonar_publisher,
                         "taluy/base_link/sonar_left_link",
                         sonar_data.distance);
    }
    if (id == KRightSonarReportIdentifier) {
      publish_sonar_data(right_sonar_publisher,
                         "taluy/base_link/sonar_right_link",
                         sonar_data.distance);
    }
  }

  bool activate_sonar(std_srvs::Trigger::Request &,
                      std_srvs::Trigger::Response &res) {
    const auto activation = []() {
      auto i = 0;
      for (i = 0; i < 4; i++) {
        i |= 1 << i;  // Set the first 4 bits to 1
      }
      return i;
    }();

    auto message = auv_canbus_msgs::SonarActivation{};
    message.flags = activation;

    dispatch_message(kSonarActivationCommand, message);

    res.success = true;
    res.message = "Sonar activation command sent successfully.";
    return true;
  }

 protected:
  void publish_sonar_data(ros::Publisher &publisher,
                          const std::string &frame_id, float distance) {
    auto sonar_msg = sensor_msgs::Range{};

    sonar_msg.header.stamp = ros::Time::now();
    sonar_msg.header.seq++;
    sonar_msg.header.frame_id = frame_id;
    sonar_msg.radiation_type = sensor_msgs::Range::ULTRASOUND;
    sonar_msg.field_of_view = 0.1;
    sonar_msg.min_range = 0.1;
    sonar_msg.max_range = 100.0;
    sonar_msg.range = distance;

    publisher.publish(sonar_msg);
  }

  bool is_sonar_id(auv_hardware::canbus::Identifier id) {
    return id == kFrontSonarReportIdentifier || id == kBackReportIdentifier ||
           id == kLeftReportIdentifier || id == KRightSonarReportIdentifier;
  }

  bool is_valid_data(const std::vector<uint8_t> &data) {
    return data.size() == sizeof(auv_canbus_msgs::DistanceMeasurement);
  }

 private:
  ros::Publisher front_sonar_publisher;
  ros::Publisher back_sonar_publisher;
  ros::Publisher left_sonar_publisher;
  ros::Publisher right_sonar_publisher;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

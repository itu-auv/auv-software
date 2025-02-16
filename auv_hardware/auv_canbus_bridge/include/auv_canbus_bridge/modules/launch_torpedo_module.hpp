#pragma once
#include <array>

#include "auv_canbus_bridge/modules/module_base.hpp"
#include "ros/ros.h"
#include "std_srvs/Trigger.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class LaunchTorpedoModule : public ModuleBase {
  constexpr static auto kLaunchTorpedo1ExtendedId =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::Mainboard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::SetHSS1OutputCommand);

  constexpr static auto kLaunchTorpedo2ExtendedId =
      auv_hardware::canbus::make_extended_id(
          0x00, auv_hardware::canbus::NodeID::Mainboard,
          auv_hardware::canbus::Function::Write,
          auv_hardware::canbus::Endpoint::SetHSS2OutputCommand);

 public:
  LaunchTorpedoModule(const ros::NodeHandle &node_handle, CanbusSocket &socket)
      : ModuleBase(node_handle, socket) {
    launch_torpedo_1_service_ = ModuleBase::node_handle().advertiseService(
        "actuators/torpedo_1/launch",
        &LaunchTorpedoModule::launch_torpedo_1_handler, this);

    launch_torpedo_2_service_ = ModuleBase::node_handle().advertiseService(
        "actuators/torpedo_2/launch",
        &LaunchTorpedoModule::launch_torpedo_2_handler, this);

    ROS_INFO_STREAM("Initialized LaunchTorpedoModule for CANBUS");
  }

  bool launch_torpedo_1_handler(std_srvs::Trigger::Request &req,
                                std_srvs::Trigger::Response &res) {
    res.success = send_hss_pulse(kLaunchTorpedo1ExtendedId);
    res.message =
        res.success ? "Torpedo 1 launched" : "Failed to launch torpedo 1";
    return true;
  }

  bool launch_torpedo_2_handler(std_srvs::Trigger::Request &req,
                                std_srvs::Trigger::Response &res) {
    res.success = send_hss_pulse(kLaunchTorpedo2ExtendedId);
    res.message =
        res.success ? "Torpedo 2 launched" : "Failed to launch torpedo 2";
    return true;
  }

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t> &data) override{
      //
  };

 protected:
  bool send_hss_pulse(auv_hardware::canbus::Identifier id) {
    bool success = true;
    std::array<uint8_t, 1> data = {1};

    success = dispatch_message(id, data);
    ros::Duration(1.0).sleep();

    data[0] = 0;
    success &= dispatch_message(id, data);

    return success;
  }

 private:
  ros::ServiceServer launch_torpedo_1_service_;
  ros::ServiceServer launch_torpedo_2_service_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "auv_canbus_bridge/canbus/helper.hpp"
#include "auv_canbus_bridge/canbus_socket.hpp"
#include "auv_canbus_bridge/modules.hpp"

namespace auv_hardware {

class CanbusBridgeROS {
 public:
  using ModuleBase = auv_hardware::canbus::modules::ModuleBase;
  using DrivePulseModule = auv_hardware::canbus::modules::DrivePulseModule;
  using LaunchTorpedoModule =
      auv_hardware::canbus::modules::LaunchTorpedoModule;
  using PowerReportModule = auv_hardware::canbus::modules::PowerReportModule;
  using KillswitchReportModule =
      auv_hardware::canbus::modules::KillswitchReportModule;

  explicit CanbusBridgeROS(const ros::NodeHandle& node_handle)
      : node_handle_{node_handle}, socket_{}, interface_name_{""} {
    auto node_handle_private = ros::NodeHandle{"~"};
    node_handle_private.param<std::string>("interface", interface_name_,
                                           "vcan0");

    socket_.initialize(interface_name_);

    modules_.reserve(4);

    modules_.emplace_back(
        std::make_unique<DrivePulseModule>(node_handle_, socket_));
    modules_.emplace_back(
        std::make_unique<LaunchTorpedoModule>(node_handle_, socket_));
    modules_.emplace_back(
        std::make_unique<PowerReportModule>(node_handle_, socket_));
    modules_.emplace_back(
        std::make_unique<KillswitchReportModule>(node_handle_, socket_));
  }

  void spin() {
    while (ros::ok()) {
      const auto message = socket_.handle();
      if (message.has_value()) {
        std::for_each(modules_.begin(), modules_.end(),
                      [&message](const std::unique_ptr<ModuleBase>& module) {
                        module->on_received_message(message->id, message->data);
                      });
      }

      ros::spinOnce();
    }
  }

 private:
  ros::NodeHandle node_handle_;
  auv_hardware::CanbusSocket socket_;
  std::string interface_name_;
  std::vector<std::unique_ptr<ModuleBase>> modules_;
};
}  // namespace auv_hardware

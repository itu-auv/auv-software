#pragma once
#include <functional>
#include <vector>

#include "auv_canbus_bridge/canbus/canbus.hpp"
#include "auv_canbus_bridge/canbus_socket.hpp"
#include "ros/node_handle.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

class ModuleBase {
 public:
  using DispatchMessageFunction =
      std::function<bool(auv_hardware::canbus::ExtendedIdType id,
                         const std::vector<uint8_t>& data)>;

  ModuleBase(const ros::NodeHandle& node_handle, CanbusSocket& socket)
      : node_handle_{node_handle}, socket_{socket} {}

  virtual void on_received_message(auv_hardware::canbus::ExtendedIdType id,
                                   const std::vector<uint8_t>& data) = 0;

  ros::NodeHandle& node_handle() { return node_handle_; }

  template <size_t N>
  bool dispatch_message(const auv_hardware::canbus::ExtendedIdType id,
                        const std::array<uint8_t, N>& data) {
    return socket_.template dispatch_message<N>(id, data);
  }

  bool dispatch_message(const auv_hardware::canbus::ExtendedIdType id,
                        const std::vector<uint8_t>& data) {
    return socket_.dispatch_message(id, data);
  }

 private:
  ros::NodeHandle node_handle_;
  CanbusSocket& socket_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware
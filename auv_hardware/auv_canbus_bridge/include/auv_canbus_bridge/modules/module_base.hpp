#pragma once
#include <ros/message_traits.h>

#include <functional>
#include <type_traits>
#include <vector>

#include "auv_canbus_bridge/canbus/canbus.hpp"
#include "auv_canbus_bridge/canbus_socket.hpp"
#include "ros/node_handle.h"

namespace auv_hardware {
namespace canbus {
namespace modules {

template <typename MessageT, typename Enable = void>
struct IsCanbusMessage : std::false_type {};

// Specialization for valid ROS messages with fixed size
template <typename MessageT>
struct IsCanbusMessage<
    MessageT, typename std::enable_if<
                  ros::message_traits::IsMessage<MessageT>::value &&
                  ros::message_traits::IsFixedSize<MessageT>::value>::type>
    : std::true_type {};

class ModuleBase {
 public:
  ModuleBase(const ros::NodeHandle& node_handle, CanbusSocket& socket)
      : node_handle_{node_handle}, socket_{socket} {}

  virtual void on_received_message(auv_hardware::canbus::Identifier id,
                                   const std::vector<uint8_t>& data) = 0;

  virtual void on_received_frame(const auv_hardware::canbus::Identifier id,
                                 const struct can_frame& frame) {
    const auto endpoint = id.endpoint();

    switch (endpoint) {
      case auv_hardware::canbus::Endpoint::FirstQuadThrusterCommand:
        /* code */
        break;

      default:
        break;
    }

    // const auto message =
    // deserialize_message<auv_msgs::MotorCommand>(frame.data, frame.can_dlc);
  }

  ros::NodeHandle& node_handle() { return node_handle_; }

  template <size_t N>
  bool dispatch_message(const auv_hardware::canbus::Identifier id,
                        const std::array<uint8_t, N>& data) {
    return false;
    // return socket_.template dispatch_message<N>(id, data);
  }

  bool dispatch_message(const auv_hardware::canbus::Identifier id,
                        const std::vector<uint8_t>& data) {
    return false;

    // return socket_.dispatch_message(id, data);
  }

  template <typename MessageT, typename = typename std::enable_if<
                                   IsCanbusMessage<MessageT>::value>::type>
  bool dispatch_message(const auv_hardware::canbus::Identifier id,
                        const MessageT& message) {
    static_assert(ros::message_traits::IsMessage<MessageT>::value,
                  "MessageT must be a ROS message");
    static_assert(ros::message_traits::IsFixedSize<MessageT>::value,
                  "MessageT must have a fixed size");

    auto lock = std::scoped_lock{outgoing_frame_mutex_};

    const auto message_size = ros::serialization::serializationLength(message);

    outgoing_frame_.can_id = id.serialized() | CAN_EFF_FLAG;
    outgoing_frame_.can_dlc = message_size;

    auto stream =
        ros::serialization::OStream{outgoing_frame_.data, message_size};

    ros::serialization::Serializer<MessageT>::write(stream, message);

    return socket_.dispatch_message(id, outgoing_frame_);
  }

  template <typename MessageT, typename = typename std::enable_if<
                                   IsCanbusMessage<MessageT>::value>::type>
  const MessageT deserialize_message(const uint8_t* data, size_t size) {
    static_assert(ros::message_traits::IsMessage<MessageT>::value,
                  "MessageT must be a ROS message");
    static_assert(ros::message_traits::IsFixedSize<MessageT>::value,
                  "MessageT must have a fixed size");

    auto stream = ros::serialization::IStream{const_cast<uint8_t*>(data), size};
    auto message = MessageT{};
    ros::serialization::Serializer<MessageT>::read(stream, message);
    return message;
  }

 private:
  ros::NodeHandle node_handle_;
  struct can_frame outgoing_frame_;
  std::mutex outgoing_frame_mutex_;
  CanbusSocket& socket_;
};

}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

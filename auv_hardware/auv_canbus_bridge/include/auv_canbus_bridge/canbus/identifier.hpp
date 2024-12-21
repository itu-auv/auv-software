#pragma once
#include "auv_canbus_bridge/canbus/endpoint.hpp"
#include "auv_canbus_bridge/canbus/function.hpp"
#include "auv_canbus_bridge/canbus/node_id.hpp"
#include "inttypes.h"
namespace auv_hardware {
namespace canbus {

class Identifier {
  static constexpr auto kPriorityMask = 0xFF000000;
  static constexpr auto kNodeIDMask = 0x00FE0000;
  static constexpr auto kFunctionMask = 0x00010000;
  static constexpr auto kEndpointMask = 0x0000FFFF;

 public:
  using SerializedExtendedIdType = uint32_t;
  using SerializedPriorityType = uint8_t;
  using SerializedNodeIdType = uint8_t;
  using SerializedFunctionType = uint8_t;
  using SerializedEndpointType = uint16_t;

  constexpr Identifier() : id_{0} {}

  constexpr Identifier(const SerializedExtendedIdType id) : id_{id} {}

  constexpr Identifier(const SerializedPriorityType priority,
                       const NodeID node_id, const Function function,
                       const Endpoint endpoint)
      : Identifier{
            serialize_identifier(priority, node_id, function, endpoint)} {}

  constexpr uint8_t priority() const {
    return static_cast<uint8_t>((id_ & kPriorityMask) >> 24);
  }

  constexpr NodeID node_id() const {
    return static_cast<NodeID>((id_ & kNodeIDMask) >> 17);
  }

  constexpr Function function() const {
    return static_cast<Function>((id_ & kFunctionMask) >> 16);
  }

  constexpr Endpoint endpoint() const {
    return static_cast<Endpoint>(id_ & kEndpointMask);
  }

  constexpr SerializedExtendedIdType serialized() const { return id_; }

  constexpr bool operator==(const Identifier &other) const {
    return id_ == other.id_;
  }

  constexpr bool operator!=(const Identifier &other) const {
    return id_ != other.id_;
  }

 private:
  static constexpr SerializedExtendedIdType serialize_identifier(
      const SerializedPriorityType priority, const NodeID node_id,
      const Function function, const Endpoint endpoint) {
    // node id is 7 bits, function is 1 bit, endpoint is 16 bits
    uint32_t identifier = 0;

    identifier |= (priority & 0xFF) << 24;
    identifier |= (static_cast<SerializedNodeIdType>(node_id) & 0x7F) << 17;
    identifier |= (static_cast<SerializedFunctionType>(function) & 0x01) << 16;
    identifier |= static_cast<SerializedEndpointType>(endpoint) & 0xFFFF;
    return identifier;
  }

  SerializedExtendedIdType id_;
};

inline constexpr Identifier make_extended_id(
    const Identifier::SerializedPriorityType priority, const NodeID node_id,
    const Function function, const Endpoint endpoint) {
  return Identifier{priority, node_id, function, endpoint};
}
}  // namespace canbus
}  // namespace auv_hardware

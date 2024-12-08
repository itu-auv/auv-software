#pragma once
#include "auv_canbus_bridge/canbus/endpoint.hpp"
#include "auv_canbus_bridge/canbus/function.hpp"
#include "auv_canbus_bridge/canbus/node_id.hpp"

namespace auv_hardware {
namespace canbus {

using ExtendedIdType = uint32_t;
using SerializedPriorityType = uint8_t;
using SerializedNodeIdType = uint8_t;
using SerializedFunctionType = uint8_t;
using SerializedEndpointType = uint16_t;

static constexpr auto kPriorityMask = 0xFF000000;
static constexpr auto kNodeIDMask = 0x00FE0000;
static constexpr auto kFunctionMask = 0x00010000;
static constexpr auto kEndpointMask = 0x0000FFFF;

inline constexpr ExtendedIdType make_extended_id(
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

inline constexpr ExtendedIdType make_extended_id(const NodeID node_id) {
  return make_extended_id(0, node_id, Function::Read,
                          Endpoint::FirstQuadThrusterCommand) &
         kNodeIDMask;
}

inline constexpr uint8_t get_priority(const ExtendedIdType extended_id) {
  return static_cast<uint8_t>((extended_id & kPriorityMask) >> 24);
}

inline constexpr NodeID get_node_id(const ExtendedIdType extended_id) {
  return static_cast<NodeID>((extended_id & kNodeIDMask) >> 17);
}

inline constexpr Function get_function(const ExtendedIdType extended_id) {
  return static_cast<Function>((extended_id & kFunctionMask) >> 16);
}

inline constexpr Endpoint get_endpoint(const ExtendedIdType extended_id) {
  return static_cast<Endpoint>(extended_id & kEndpointMask);
}

}  // namespace canbus
}  // namespace auv_hardware
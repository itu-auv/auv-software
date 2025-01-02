#pragma once
#include "inttypes.h"

namespace auv_hardware {
namespace canbus {

enum class NodeID : uint8_t {
  PropulsionBoard = 0,
  Mainboard = 1,
  ExpansionBoard = 2,
  ROSBridge = 3,
};

}  // namespace canbus
}  // namespace auv_hardware

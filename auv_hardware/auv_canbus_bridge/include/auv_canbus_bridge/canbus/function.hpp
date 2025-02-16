#pragma once
#include "inttypes.h"

namespace auv_hardware {
namespace canbus {

enum class Function : uint8_t {
  Read = 0,
  Write = 1,
};

}  // namespace canbus
}  // namespace auv_hardware

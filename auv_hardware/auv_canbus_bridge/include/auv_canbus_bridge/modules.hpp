#pragma once

#include <variant>

#include "auv_canbus_bridge/modules/drive_pulse_module.hpp"
#include "auv_canbus_bridge/modules/imu_module.hpp"
#include "auv_canbus_bridge/modules/killswitch_report_module.hpp"
#include "auv_canbus_bridge/modules/launch_torpedo_module.hpp"
#include "auv_canbus_bridge/modules/marker_dropper_module.hpp"
#include "auv_canbus_bridge/modules/ping_sonar_module.hpp"
#include "auv_canbus_bridge/modules/power_report_module.hpp"
#include "auv_canbus_bridge/modules/pressure_report_module.hpp"

namespace auv_hardware {
namespace canbus {
namespace modules {}  // namespace modules
}  // namespace canbus
}  // namespace auv_hardware

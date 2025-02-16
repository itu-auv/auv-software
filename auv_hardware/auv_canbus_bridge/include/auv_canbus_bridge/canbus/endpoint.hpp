#pragma once
#include "inttypes.h"

namespace auv_hardware {
namespace canbus {

enum class Endpoint : uint16_t {
  FirstQuadThrusterCommand = 0,
  SecondQuadThrusterCommand,
  SetHSS1OutputCommand,
  SetHSS2OutputCommand,
  SetHSS3OutputCommand,
  LatchedServoCommand,
  SonarActivationCommand,
  MainboardPowerReport,
  BackSonarReport,
  FrontSonarReport,
  RightSonarReport,
  LeftSonarReport,
  PropulsionBoardStatusReport,
  Count
};
}  // namespace canbus
}  // namespace auv_hardware

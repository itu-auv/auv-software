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
  MainboardPowerReport,
  BackSonarReport,
  FrontSonarReport,
  RightSonarReport,
  LeftSonarReport,
  PropulsionBoardStatusReport,
  SonarActivationCommand,
  OrientationReport,
  AngularVelocityReport,
  LinearAccelerationReport,
  PressureSensorReport,
  Count
};
}  // namespace canbus
}  // namespace auv_hardware

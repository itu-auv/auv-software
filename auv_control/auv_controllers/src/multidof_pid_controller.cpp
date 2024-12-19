#include "auv_controllers/multidof_pid_controller.h"

#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(auv::control::SixDOFPIDController,
                       auv::control::SixDOFControllerBase)

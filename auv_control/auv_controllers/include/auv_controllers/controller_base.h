#pragma once
#include <Eigen/Core>

#include "auv_common_lib/control/controller_types.h"
#include "auv_controllers/model.h"

namespace auv {
namespace control {

class SixDOFControllerBase {
 public:
  using Matrix = Matrix6d;
  using Vector = Vector6d;
  using WrenchVector = Vector;

  virtual ~SixDOFControllerBase() = default;

  virtual WrenchVector control(const SixDOFState& state,
                               const SixDOFCommand& command,
                               const SixDOFStateDerivative& state_derivative,
                               const double dt) = 0;

  virtual void set_parameters(const SixDOFControllerParameters& parameters) {
    parameters_ = parameters;
  }

  const SixDOFControllerParameters& parameters() const { return parameters_; }

  const SixDOFModel& model() const { return model_; }

  void set_model(const SixDOFModel& model) { model_ = model; }

 protected:
  SixDOFModel model_;
  SixDOFControllerParameters parameters_;
};

}  // namespace control
}  // namespace auv

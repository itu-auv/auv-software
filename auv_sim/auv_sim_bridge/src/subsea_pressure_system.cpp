#include <gz/msgs/fluid_pressure.pb.h>

#include <chrono>
#include <cmath>
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/transport/Node.hh>
#include <random>
#include <string>

namespace auv_sim_bridge {

// Harmonic adapter for the legacy UUV SubseaPressureROSPlugin measurement.
// The sensor reads physics pose directly; it does not subscribe to odometry.
class SubseaPressureSystem : public gz::sim::System,
                             public gz::sim::ISystemConfigure,
                             public gz::sim::ISystemPostUpdate {
 public:
  void Configure(const gz::sim::Entity& entity,
                 const std::shared_ptr<const sdf::Element>& sdf,
                 gz::sim::EntityComponentManager&,
                 gz::sim::EventManager&) override {
    model_ = entity;
    link_name_ = sdf->Get<std::string>("link_name");
    frame_ = sdf->Get<std::string>("frame_id");
    offset_ = sdf->Get<gz::math::Pose3d>("sensor_pose");
    const auto rate = sdf->Get<double>("update_rate", 20.0).first;
    period_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / rate));
    standard_pressure_ = sdf->Get<double>("standard_pressure", 101.325).first;
    kpa_per_meter_ = sdf->Get<double>("kPa_per_meter", 9.80638).first;
    noise_sigma_ = sdf->Get<double>("noise_sigma", 0.4).first;
    noise_amplitude_ = sdf->Get<double>("noise_amplitude", 0.13).first;
    noise_ = std::normal_distribution<double>(0.0, noise_sigma_);
    publisher_ = node_.Advertise<gz::msgs::FluidPressure>(
        sdf->Get<std::string>("topic"));
  }

  void PostUpdate(const gz::sim::UpdateInfo& info,
                  const gz::sim::EntityComponentManager& ecm) override {
    if (info.paused || info.simTime - last_update_ < period_) return;
    if (link_ == gz::sim::kNullEntity) {
      link_ = ecm.EntityByComponents(gz::sim::components::Link(),
                                     gz::sim::components::Name(link_name_),
                                     gz::sim::components::ParentEntity(model_));
      if (link_ == gz::sim::kNullEntity) return;
    }
    const auto sensor_pose = gz::sim::worldPose(link_, ecm) * offset_;
    // Preserve UUV's abs(Z), kPa/m coefficient and Gaussian amplitude.
    const double pressure_kpa =
        standard_pressure_ + std::abs(sensor_pose.Pos().Z()) * kpa_per_meter_ +
        noise_amplitude_ * noise_(random_);
    gz::msgs::FluidPressure message;
    const auto nanoseconds =
        std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime)
            .count();
    message.mutable_header()->mutable_stamp()->set_sec(nanoseconds /
                                                       1000000000);
    message.mutable_header()->mutable_stamp()->set_nsec(nanoseconds %
                                                        1000000000);
    auto* frame = message.mutable_header()->add_data();
    frame->set_key("frame_id");
    frame->add_value(frame_);
    // Gazebo / ROS 2 FluidPressure uses Pa. Legacy UUV published kPa.
    message.set_pressure(1000.0 * pressure_kpa);
    message.set_variance(1000000.0 * noise_sigma_ * noise_sigma_);
    publisher_.Publish(message);
    last_update_ = info.simTime;
  }

 private:
  gz::sim::Entity model_{gz::sim::kNullEntity};
  gz::sim::Entity link_{gz::sim::kNullEntity};
  std::string link_name_, frame_;
  gz::math::Pose3d offset_;
  std::chrono::steady_clock::duration period_{}, last_update_{};
  double standard_pressure_{101.325}, kpa_per_meter_{9.80638};
  double noise_sigma_{0.4}, noise_amplitude_{0.13};
  std::default_random_engine random_{std::random_device{}()};
  std::normal_distribution<double> noise_;
  gz::transport::Node node_;
  gz::transport::Node::Publisher publisher_;
};

}  // namespace auv_sim_bridge

GZ_ADD_PLUGIN(auv_sim_bridge::SubseaPressureSystem, gz::sim::System,
              auv_sim_bridge::SubseaPressureSystem::ISystemConfigure,
              auv_sim_bridge::SubseaPressureSystem::ISystemPostUpdate)

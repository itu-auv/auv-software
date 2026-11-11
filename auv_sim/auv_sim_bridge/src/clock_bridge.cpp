// Simulation time for ROS, at a chosen rate. Gazebo publishes its clock every
// physics step; every rclpy node wakes up for each tick, and with forty Python
// nodes a fast clock alone saturates the machine. This forwards a tick every
// `period` seconds of simulation time instead.
#include <gz/msgs/clock.pb.h>

#include <atomic>
#include <chrono>
#include <gz/transport/Node.hh>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rosgraph_msgs/msg/clock.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("simulation_clock_bridge");
  const auto period = rclcpp::Duration::from_seconds(
      node->declare_parameter<double>("period", 0.01));
  auto publisher = node->create_publisher<rosgraph_msgs::msg::Clock>(
      "/clock", rclcpp::ClockQoS());

  rclcpp::Time last{0, 0, RCL_ROS_TIME};
  std::atomic<std::chrono::steady_clock::rep> last_received{
      std::chrono::steady_clock::now().time_since_epoch().count()};
  std::atomic<bool> received_any{false};
  gz::transport::Node gz_node;
  std::function<void(const gz::msgs::Clock&)> callback =
      [&](const gz::msgs::Clock& msg) {
        last_received =
            std::chrono::steady_clock::now().time_since_epoch().count();
        received_any = true;
        const rclcpp::Time now{static_cast<int32_t>(msg.sim().sec()),
                               static_cast<uint32_t>(msg.sim().nsec()),
                               RCL_ROS_TIME};
        if (now < last || now - last >= period) {
          rosgraph_msgs::msg::Clock out;
          out.clock = now;
          publisher->publish(out);
          last = now;
        }
      };
  if (!gz_node.Subscribe("/clock", callback)) {
    RCLCPP_FATAL(node->get_logger(), "Cannot subscribe to the Gazebo clock");
    return 1;
  }
  // gz-transport (Jetty) sometimes stops delivering to a live subscription,
  // and then every sim-time timer in ROS stops. Subscribe again when it does.
  auto watchdog = node->create_wall_timer(std::chrono::seconds(1), [&]() {
    const auto quiet =
        std::chrono::steady_clock::now() -
        std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(last_received.load()));
    if (quiet < std::chrono::seconds(2)) return;
    if (received_any) {  // before that, Gazebo is still loading the world
      RCLCPP_WARN(node->get_logger(),
                  "No Gazebo clock for %.1f s, subscribing again",
                  std::chrono::duration<double>(quiet).count());
    }
    gz_node.Unsubscribe("/clock");
    gz_node.Subscribe("/clock", callback);
    last_received = std::chrono::steady_clock::now().time_since_epoch().count();
  });
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

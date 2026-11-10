// Simulation time for ROS, at a chosen rate. Gazebo publishes its clock every
// physics step (1 kHz); every rclpy node wakes up for each tick, and with forty
// Python nodes that alone saturates the machine. This forwards a tick every
// `period` seconds of simulation time instead.
#include <gz/msgs/clock.pb.h>

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
  gz::transport::Node gz_node;
  std::function<void(const gz::msgs::Clock&)> callback =
      [&](const gz::msgs::Clock& msg) {
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
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

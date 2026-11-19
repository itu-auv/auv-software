// Gazebo -> ROS for sensor streams, as ros_gz_bridge's parameter_bridge does,
// but a stream that goes quiet is subscribed again. On Jetty, Gazebo sometimes
// loses a subscriber's registration and then sends it nothing, with the
// connection up. Quiet is measured in simulation time: a slow Gazebo is not a
// lost subscription, and resubscribing everything would slow it further.
//   topics: ["<gz topic> <ros topic> <type>", ...]
//   type: image | camera_info | imu | odometry | dvl | fluid_pressure
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <gz/transport/Node.hh>
#include <marine_acoustic_msgs/msg/dvl.hpp>
#include <memory>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <ros_gz_bridge/convert.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/fluid_pressure.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Stream {
  std::string gz_topic;
  std::function<void()> subscribe;
  std::atomic<int64_t> last_received{0};  // node time, nanoseconds
  std::atomic<bool> received_any{false};
};

template <class GzMsg, class RosMsg>
std::unique_ptr<Stream> make_stream(rclcpp::Node& node,
                                    gz::transport::Node& gz_node,
                                    const std::string& gz_topic,
                                    const std::string& ros_topic) {
  auto stream = std::make_unique<Stream>();
  stream->gz_topic = gz_topic;
  std::vector<typename rclcpp::Publisher<RosMsg>::SharedPtr> publishers;
  std::istringstream names(
      ros_topic);  // a,b: the same stream on several topics
  for (std::string name; std::getline(names, name, ',');) {
    publishers.push_back(node.create_publisher<RosMsg>(name, rclcpp::QoS(10)));
  }
  Stream* s = stream.get();
  std::function<void(const GzMsg&)> callback = [s, publishers,
                                                &node](const GzMsg& in) {
    s->last_received = node.now().nanoseconds();
    s->received_any = true;
    RosMsg out;
    ros_gz_bridge::convert_gz_to_ros(in, out);
    for (const auto& publisher : publishers) publisher->publish(out);
  };
  stream->subscribe = [&gz_node, gz_topic, callback, &node]() {
    if (!gz_node.Subscribe(gz_topic, callback)) {
      RCLCPP_ERROR(node.get_logger(), "Cannot subscribe to %s",
                   gz_topic.c_str());
    }
  };
  return stream;
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("gz_sensor_bridge");
  const auto topics = node->declare_parameter<std::vector<std::string>>(
      "topics", std::vector<std::string>{});
  const auto silence = node->declare_parameter<double>("silence", 2.0);

  gz::transport::Node gz_node;
  std::vector<std::unique_ptr<Stream>> streams;
  for (const auto& spec : topics) {
    std::istringstream fields(spec);
    std::string gz_topic, ros_topic, type;
    fields >> gz_topic >> ros_topic >> type;
    std::unique_ptr<Stream> stream;
    if (type == "image") {
      stream = make_stream<gz::msgs::Image, sensor_msgs::msg::Image>(
          *node, gz_node, gz_topic, ros_topic);
    } else if (type == "camera_info") {
      stream = make_stream<gz::msgs::CameraInfo, sensor_msgs::msg::CameraInfo>(
          *node, gz_node, gz_topic, ros_topic);
    } else if (type == "imu") {
      stream = make_stream<gz::msgs::IMU, sensor_msgs::msg::Imu>(
          *node, gz_node, gz_topic, ros_topic);
    } else if (type == "odometry") {
      stream = make_stream<gz::msgs::Odometry, nav_msgs::msg::Odometry>(
          *node, gz_node, gz_topic, ros_topic);
    } else if (type == "dvl") {
      stream = make_stream<gz::msgs::DVLVelocityTracking,
                           marine_acoustic_msgs::msg::Dvl>(*node, gz_node,
                                                           gz_topic, ros_topic);
    } else if (type == "fluid_pressure") {
      stream =
          make_stream<gz::msgs::FluidPressure, sensor_msgs::msg::FluidPressure>(
              *node, gz_node, gz_topic, ros_topic);
    } else {
      RCLCPP_FATAL(node->get_logger(), "Unknown stream '%s'", spec.c_str());
      return 1;
    }
    stream->subscribe();
    streams.push_back(std::move(stream));
  }

  auto watchdog = node->create_wall_timer(std::chrono::seconds(1), [&]() {
    const auto now = node->now().nanoseconds();
    for (auto& s : streams) {
      const auto quiet = 1e-9 * static_cast<double>(now - s->last_received);
      if (quiet < silence) continue;
      if (s->received_any) {  // before that, Gazebo may still be starting
        RCLCPP_WARN(node->get_logger(),
                    "%s quiet for %.1f s, subscribing again",
                    s->gz_topic.c_str(), quiet);
      }
      gz_node.Unsubscribe(s->gz_topic);
      s->subscribe();
      s->last_received = now;
    }
  });

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

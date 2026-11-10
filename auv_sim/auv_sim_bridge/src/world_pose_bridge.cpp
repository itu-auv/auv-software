// World entity poses for ROS, with their names (what /gazebo/model_states was in
// Gazebo Classic). ros_gz_bridge converts gz.msgs.Pose_V to TFMessage without the
// entity names, so this node does it: child_frame_id is the entity name.
#include <memory>
#include <string>

#include <gz/msgs/pose_v.pb.h>
#include <gz/transport/Node.hh>
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("world_pose_bridge");
  const auto world = node->declare_parameter<std::string>("world", "pool");
  const auto topic = "/world/" + world + "/pose/info";
  // Unpacking sixty transforms per message is slow in Python; 20 Hz of simulation
  // time is plenty for projecting objects into camera images.
  const auto period = node->declare_parameter<double>("period", 0.05);
  auto publisher = node->create_publisher<tf2_msgs::msg::TFMessage>(topic, 1);
  double last = -1.0;

  gz::transport::Node gz_node;
  std::function<void(const gz::msgs::Pose_V&)> callback =
      [&](const gz::msgs::Pose_V& poses) {
        const double sim_time = poses.header().stamp().sec() +
                                1e-9 * poses.header().stamp().nsec();
        if (sim_time >= last && sim_time - last < period) {
          return;
        }
        last = sim_time;
        tf2_msgs::msg::TFMessage out;
        out.transforms.reserve(poses.pose_size());
        const auto stamp = node->now();
        for (const auto& pose : poses.pose()) {
          geometry_msgs::msg::TransformStamped transform;
          transform.header.stamp = stamp;
          transform.header.frame_id = world;
          transform.child_frame_id = pose.name();
          transform.transform.translation.x = pose.position().x();
          transform.transform.translation.y = pose.position().y();
          transform.transform.translation.z = pose.position().z();
          transform.transform.rotation.x = pose.orientation().x();
          transform.transform.rotation.y = pose.orientation().y();
          transform.transform.rotation.z = pose.orientation().z();
          transform.transform.rotation.w = pose.orientation().w();
          out.transforms.push_back(std::move(transform));
        }
        publisher->publish(out);
      };
  if (!gz_node.Subscribe(topic, callback)) {
    RCLCPP_FATAL(node->get_logger(), "Cannot subscribe to Gazebo topic %s",
                 topic.c_str());
    return 1;
  }
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

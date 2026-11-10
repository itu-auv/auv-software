#pragma once

#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "auv_mapping/object_position_filter.hpp"
#include "auv_msgs/srv/set_object_transform.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

namespace auv_mapping {

class ObjectMapTFServerROS {
  using FilterMap =
      std::unordered_map<std::string, std::vector<ObjectPositionFilter::Ptr>>;

 public:
  ObjectMapTFServerROS(const rclcpp::Node::SharedPtr &nh);

  void run();

 private:
  void broadcast_transforms();

  bool clear_map_handler(std_srvs::srv::Trigger::Request &req,
                         std_srvs::srv::Trigger::Response &res);

  bool set_transform_handler(auv_msgs::srv::SetObjectTransform::Request &req,
                             auv_msgs::srv::SetObjectTransform::Response &res);

  void dynamic_transform_callback(
      const geometry_msgs::msg::TransformStamped::ConstSharedPtr &msg);
  void update_filter_frame_index(const std::string &object_frame);
  std::optional<geometry_msgs::msg::TransformStamped> transform_to_static_frame(
      const geometry_msgs::msg::TransformStamped transform);

  std::optional<geometry_msgs::msg::TransformStamped> get_transform(
      const std::string &target_frame, const std::string &source_frame,
      const rclcpp::Time &lookup_time, const rclcpp::Duration timeout);

  rclcpp::Node::SharedPtr nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  double rate_;
  double distance_threshold_squared_;
  std::string static_frame_;
  std::string base_link_frame_;
  FilterMap filters_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  //
  std::mutex mutex_;
  rclcpp::Service<auv_msgs::srv::SetObjectTransform>::SharedPtr service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_service_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr
      dynamic_sub_;
};

}  // namespace auv_mapping

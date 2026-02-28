#pragma once

#include <cmath>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "auv_mapping/object_position_filter.hpp"
#include "auv_msgs/SetObjectTransform.h"
#include "geometry_msgs/TransformStamped.h"
#include "ros/ros.h"
#include "std_srvs/Trigger.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"

namespace auv_mapping {

class ObjectMapTFServerROS {
  using FilterMap =
      std::unordered_map<std::string, std::vector<ObjectPositionFilter::Ptr>>;

 public:
  ObjectMapTFServerROS(const ros::NodeHandle &nh);

  void run();

 private:
  void broadcast_transforms();

  bool clear_map_handler(std_srvs::Trigger::Request &req,
                         std_srvs::Trigger::Response &res);

  bool set_transform_handler(auv_msgs::SetObjectTransform::Request &req,
                             auv_msgs::SetObjectTransform::Response &res);

  void dynamic_transform_callback(
      const geometry_msgs::TransformStamped::ConstPtr &msg);
  void update_filter_frame_index(const std::string &object_frame);
  std::optional<geometry_msgs::TransformStamped> transform_to_static_frame(
      const geometry_msgs::TransformStamped transform);

  std::optional<geometry_msgs::TransformStamped> get_transform(
      const std::string &target_frame, const std::string &source_frame,
      const ros::Duration timeout);

  ros::NodeHandle nh_;
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
  ros::ServiceServer service_;
  ros::ServiceServer clear_service_;
  ros::Subscriber dynamic_sub_;
};

}  // namespace auv_mapping

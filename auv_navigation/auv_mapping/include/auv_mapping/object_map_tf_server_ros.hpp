#pragma once

#include <auv_msgs/SetObjectTransform.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <mutex>
#include <optional>
#include <unordered_map>

namespace auv_mapping {
class ObjectMapTFServerROS {
  using TransformMap =
      std::unordered_map<std::string, geometry_msgs::TransformStamped>;

 public:
  ObjectMapTFServerROS(const ros::NodeHandle &nh)
      : nh_{nh},
        tf_buffer_{},
        tf_listener_{tf_buffer_},
        rate_{10.0},
        static_frame_{""},
        transforms_{},
        tf_broadcaster_{} {
    auto node_handler_private = ros::NodeHandle{"~"};

    node_handler_private.param<std::string>("static_frame", static_frame_,
                                            "odom");
    node_handler_private.param<double>("rate", rate_, 10.0);

    service_ = nh_.advertiseService(
        "set_object_transform", &ObjectMapTFServerROS::set_transform_handler,
        this);

    ROS_INFO("ObjectMapTFServerROS initialized. Static frame: %s",
             static_frame_.c_str());
  }

  bool set_transform_handler(auv_msgs::SetObjectTransform::Request &req,
                             auv_msgs::SetObjectTransform::Response &res) {
    const auto parent_frame = req.transform.header.frame_id;
    const auto target_frame = req.transform.child_frame_id;
    const auto static_to_parent_transform =
        get_transform(static_frame_, parent_frame, ros::Duration(4.0));

    if (!static_to_parent_transform.has_value()) {
      ROS_ERROR("Error occurred while looking up transform:(ozanhakantunca)");
      res.success = false;
      res.message = "Failed to capture transform TODO (ozanhakantunca)";
      return false;
    }

    const auto parent_to_target_quaternion = tf2::Quaternion(
        req.transform.transform.rotation.x, req.transform.transform.rotation.y,
        req.transform.transform.rotation.z, req.transform.transform.rotation.w);

    auto parent_to_target_orientation =
        tf2::Matrix3x3{parent_to_target_quaternion};

    auto parent_to_target_translation =
        tf2::Vector3{req.transform.transform.translation.x,
                     req.transform.transform.translation.y,
                     req.transform.transform.translation.z};

    auto static_to_parent_quaternion = tf2::Quaternion{};
    tf2::fromMsg(static_to_parent_transform->transform.rotation,
                 static_to_parent_quaternion);

    const auto static_to_parent_orientation =
        tf2::Matrix3x3{static_to_parent_quaternion};

    const auto static_to_parent_translation =
        tf2::Vector3{static_to_parent_transform->transform.translation.x,
                     static_to_parent_transform->transform.translation.y,
                     static_to_parent_transform->transform.translation.z};

    const auto static_to_target_orientation = tf2::Matrix3x3{
        static_to_parent_orientation * parent_to_target_orientation};

    const auto static_to_target_translation = tf2::Vector3{
        static_to_parent_translation +
        static_to_parent_orientation * parent_to_target_translation};

    auto static_transform = geometry_msgs::TransformStamped{};
    static_transform.header.stamp = ros::Time::now();
    static_transform.header.frame_id = static_frame_;
    static_transform.child_frame_id = target_frame;

    static_transform.transform.translation.x = static_to_target_translation.x();
    static_transform.transform.translation.y = static_to_target_translation.y();
    static_transform.transform.translation.z = static_to_target_translation.z();

    auto static_to_target_quaternion = tf2::Quaternion{};
    static_to_target_orientation.getRotation(static_to_target_quaternion);
    static_transform.transform.rotation =
        tf2::toMsg(static_to_target_quaternion);

    {
      auto lock = std::scoped_lock(mutex_);
      transforms_[target_frame] = static_transform;
    }

    ROS_INFO_STREAM("Stored static transform from " << static_frame_ << " to "
                                                    << target_frame);
    res.success = true;
    res.message = "Stored transform for frame: " + target_frame;
    return true;
  }

  void publishTransforms() {
    auto rate = ros::Rate{rate_};
    while (ros::ok()) {
      {
        auto lock = std::scoped_lock(mutex_);

        for (const auto &entry : transforms_) {
          auto transform = entry.second;
          transform.header.stamp = ros::Time::now();
          tf_broadcaster_.sendTransform(transform);
        }
      }
      rate.sleep();
    }
  }

 private:
  std::optional<geometry_msgs::TransformStamped> get_transform(
      const std::string &target_frame, const std::string &source_frame,
      const ros::Duration timeout) {
    try {
      auto transform = tf_buffer_.lookupTransform(target_frame, source_frame,
                                                  ros::Time(0), timeout);
      return transform;
    } catch (tf2::TransformException &ex) {
      ROS_WARN_STREAM("Transform lookup failed: " << ex.what());
      return std::nullopt;
    }
  }

  ros::NodeHandle nh_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  double rate_;
  std::string static_frame_;
  TransformMap transforms_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  //
  std::mutex mutex_;
  ros::ServiceServer service_;
};

}  // namespace auv_mapping

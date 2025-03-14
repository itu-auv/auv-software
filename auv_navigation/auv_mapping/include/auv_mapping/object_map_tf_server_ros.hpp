#pragma once

#include <auv_msgs/SetObjectTransform.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <cmath>
#include <mutex>
#include <optional>
#include <string>
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

    // Service to "lock in" or reset an object's transform.
    service_ = nh_.advertiseService(
        "set_object_transform", &ObjectMapTFServerROS::set_transform_handler,
        this);

    dynamic_sub_ =
        nh_.subscribe("object_transform_updates", 10,
                      &ObjectMapTFServerROS::dynamic_transform_callback, this);

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
      ROS_ERROR("Error occurred while looking up transform");
      res.success = false;
      res.message = "Failed to capture transform";
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

  void dynamic_transform_callback(
      const geometry_msgs::TransformStamped::ConstPtr &msg) {
    std::scoped_lock lock(mutex_);
    std::string base_frame = msg->child_frame_id;
    auto it = transforms_.find(base_frame);
    if (it == transforms_.end()) {
      transforms_[base_frame] = *msg;
      ROS_INFO_STREAM("Added new dynamic transform for " << base_frame);
    } else {
      const auto &current_transform = it->second;
      double dx = current_transform.transform.translation.x -
                  msg->transform.translation.x;
      double dy = current_transform.transform.translation.y -
                  msg->transform.translation.y;
      double dz = current_transform.transform.translation.z -
                  msg->transform.translation.z;
      double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

      const double DIST_THRESHOLD = 1.0;
      if (distance < DIST_THRESHOLD) {
        double alpha = 0.2;
        transforms_[base_frame].transform.translation.x =
            alpha * msg->transform.translation.x +
            (1 - alpha) * current_transform.transform.translation.x;
        transforms_[base_frame].transform.translation.y =
            alpha * msg->transform.translation.y +
            (1 - alpha) * current_transform.transform.translation.y;
        transforms_[base_frame].transform.translation.z =
            alpha * msg->transform.translation.z +
            (1 - alpha) * current_transform.transform.translation.z;
        transforms_[base_frame].transform.rotation = msg->transform.rotation;
        ROS_INFO_STREAM("Updated dynamic transform for " << base_frame);
      } else {
        int suffix = 0;
        std::string new_frame;
        do {
          new_frame = base_frame + "_" + std::to_string(suffix++);
        } while (transforms_.find(new_frame) != transforms_.end());
        geometry_msgs::TransformStamped new_transform = *msg;
        new_transform.child_frame_id = new_frame;
        transforms_[new_frame] = new_transform;
        ROS_INFO_STREAM("Created new dynamic transform for "
                        << new_frame << " due to large distance (" << distance
                        << ")");
      }
    }
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
  ros::Subscriber dynamic_sub_;
};

}  // namespace auv_mapping

#include "auv_mapping/object_map_tf_server_ros.hpp"

#include <numeric>

// #include <auv_msgs/SetObjectTransform.h>
// #include <geometry_msgs/TransformStamped.h>
// #include <ros/ros.h>
// #include <std_srvs/Empty.h>
// #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
// #include <tf2_ros/transform_broadcaster.h>
// #include <tf2_ros/transform_listener.h>

// #include <cmath>
// #include <memory>
// #include <mutex>
// #include <numeric>
// #include <optional>
// #include <string>
// #include <unordered_map>
// #include <vector>

// #include "auv_mapping/object_position_filter.hpp"

namespace auv_mapping {

ObjectMapTFServerROS::ObjectMapTFServerROS(const ros::NodeHandle& nh)
    : nh_{nh},
      tf_buffer_{ros::Duration(60.0)},
      tf_listener_{tf_buffer_},
      rate_{10.0},
      distance_threshold_squared_{16.0},
      static_frame_{""},
      tf_broadcaster_{} {
  auto node_handler_private = ros::NodeHandle{"~"};

  node_handler_private.param<std::string>("static_frame", static_frame_,
                                          "odom");
  node_handler_private.param<double>("rate", rate_, 10.0);

  double distance_threshold_;
  node_handler_private.param<double>("distance_threshold", distance_threshold_,
                                     4.0);  // 4.0
  distance_threshold_squared_ = std::pow(distance_threshold_, 2);

  service_ =
      nh_.advertiseService("set_object_transform",
                           &ObjectMapTFServerROS::set_transform_handler, this);

  clear_service_ =
      nh_.advertiseService("clear_object_transforms",
                           &ObjectMapTFServerROS::clear_map_handler, this);

  dynamic_sub_ =
      nh_.subscribe("object_transform_updates", 10,
                    &ObjectMapTFServerROS::dynamic_transform_callback, this);

  object_transform_non_kalman_create_sub_ = nh_.subscribe(
      "object_transform_non_kalman_create", 10,
      &ObjectMapTFServerROS::object_transform_non_kalman_create_callback, this);

  ROS_DEBUG("ObjectMapTFServerROS initialized. Static frame: %s",
            static_frame_.c_str());
}

void ObjectMapTFServerROS::run() {
  auto rate = ros::Rate{rate_};
  while (ros::ok()) {
    broadcast_transforms();
    broadcast_non_kalman_transforms();
    rate.sleep();
  }
}

void ObjectMapTFServerROS::broadcast_transforms() {
  auto lock = std::scoped_lock{mutex_};
  for (auto& entry : filters_) {
    for (const auto& filter_ptr : entry.second) {
      tf_broadcaster_.sendTransform(filter_ptr->getFilteredTransform());
    }
  }
}

void ObjectMapTFServerROS::broadcast_non_kalman_transforms() {
  auto lock = std::scoped_lock{mutex_};
  for (auto& entry : non_kalman_filters_) {
    for (auto& transform : entry.second) {
      transform.header.stamp = ros::Time::now();
      tf_broadcaster_.sendTransform(transform);
    }
  }
}

void ObjectMapTFServerROS::object_transform_non_kalman_create_callback(
    const geometry_msgs::TransformStamped::ConstPtr& msg) {
  const auto static_transform = transform_to_static_frame(*msg);

  if (!static_transform.has_value()) {
    ROS_ERROR("Failed to capture transform");
    return;
  }

  const auto target_frame = msg->child_frame_id;

  {
    auto lock = std::scoped_lock{mutex_};
    auto it = non_kalman_filters_.find(target_frame);
    if (it != non_kalman_filters_.end()) {
      non_kalman_filters_[target_frame].clear();
    }

    non_kalman_filters_[target_frame].push_back(*static_transform);
  }

  ROS_DEBUG_STREAM("Stored static transform from " << static_frame_ << " to "
                                                   << target_frame);
}

bool ObjectMapTFServerROS::clear_map_handler(std_srvs::Trigger::Request& req,
                                             std_srvs::Trigger::Response& res) {
  auto lock = std::scoped_lock{mutex_};
  filters_.clear();

  res.success = true;
  res.message = "Cleared all object transforms and filters.";

  ROS_INFO("Cleared all object transforms and filters.");
  return true;
}

bool ObjectMapTFServerROS::set_transform_handler(
    auv_msgs::SetObjectTransform::Request& req,
    auv_msgs::SetObjectTransform::Response& res) {
  const auto static_transform = transform_to_static_frame(req.transform);

  if (!static_transform.has_value()) {
    res.success = false;
    res.message = "Failed to capture transform";
    return false;
  }

  const auto target_frame = req.transform.child_frame_id;

  // Ben daha önce flters_ içinde olan nesneyi silmek ve onun yerine yenisini
  // eklemek istiyorum. Ama bunu yaparken eğer işlem aynı anda olursa
  // segmentation fault hatası alıyorum. Bu yüzden mutex kullanıyorum.
  // scoped_lock ile mutex kilitleniyor ve köşeli parantezden çıkınca otomatik
  // açılıyor eğer sadece lock kullanırsak bu esnada bir hata olması durumunda
  // sistem sonsuza kadar kilitli kalabilir
  {
    auto lock = std::scoped_lock{mutex_};
    auto it = filters_.find(target_frame);
    if (it != filters_.end()) {
      filters_[target_frame].clear();
    }

    filters_[target_frame].push_back(
        std::make_unique<ObjectPositionFilter>(*static_transform, 1.0 / rate_));
  }

  ROS_DEBUG_STREAM("Stored static transform from " << static_frame_ << " to "
                                                   << target_frame);
  res.success = true;
  res.message = "Stored transform for frame: " + target_frame;
  return true;
}

void ObjectMapTFServerROS::dynamic_transform_callback(
    const geometry_msgs::TransformStamped::ConstPtr& msg) {
  auto lock = std::scoped_lock{mutex_};

  // Calculate actual dt
  static auto last_time = ros::Time::now();
  const auto current_time = ros::Time::now();
  const auto dt = (current_time - last_time).toSec();
  last_time = current_time;

  const auto object_frame = msg->child_frame_id;
  const auto static_transform =
      transform_to_static_frame(*msg);  // odom -> childe
  if (!static_transform.has_value()) {
    ROS_ERROR("Failed to capture transform");
    return;
  }

  auto it = filters_.find(object_frame);
  if (it == filters_.end()) {  // eğer bu nesneye ait bir filter yoksa
    // Create first filter for this object
    filters_[object_frame].push_back(
        std::make_unique<ObjectPositionFilter>(*static_transform, 1.0 / rate_));
    // eğer filters_ içinde bu id'li frame'in atandığı bir kalman filtresi yoksa
    // yeni bir tane oluşturuyoruz
    ROS_DEBUG_STREAM("Created new filter for " << object_frame);
    return;
  }

  bool filter_updated = false;
  const bool is_slalom_gate =
      object_frame.find("red_pipe_link") != std::string::npos ||
      object_frame.find("white_pipe_link") != std::string::npos;

  // If object is a slalom gate, use a smaller distance threshold
  double current_distance_threshold_squared = distance_threshold_squared_;
  if (is_slalom_gate) {
    current_distance_threshold_squared = 1.0;  // 1.0 metre'nin karesi
    ROS_DEBUG_STREAM(
        "Using special distance threshold for slalom gate: " << object_frame);
  }

  // Find the closest filter to update
  std::vector<double> distances;
  distances.reserve(it->second.size());

  // Calculate distance to each existing filter
  for (auto& filter_ptr : it->second) {
    const auto current = filter_ptr->getFilteredTransform();
    const auto d_position =
        std::array<double, 3>{current.transform.translation.x -
                                  static_transform->transform.translation.x,
                              current.transform.translation.y -
                                  static_transform->transform.translation.y,
                              current.transform.translation.z -
                                  static_transform->transform.translation.z};
    const auto distance_squared = std::inner_product(
        d_position.begin(), d_position.end(), d_position.begin(), 0.0);
    distances.push_back(distance_squared);

    // If this filter is close enough, update it
    if (distance_squared < current_distance_threshold_squared) {
      filter_ptr->update(*static_transform, dt);
      filter_updated = true;
      ROS_DEBUG_STREAM("Updated filter for " << object_frame);

      // For non-slalom objects, update only the first matching filter.
      // For slalom gates, continue to update all filters within the threshold.
      if (!is_slalom_gate) {
        break;
      }
    }
  }

  // If no filter was updated, create a new one
  if (!filter_updated) {
    filters_[object_frame].push_back(
        std::make_unique<ObjectPositionFilter>(*static_transform, 1.0 / rate_));
    ROS_DEBUG_STREAM("Created new filter for "
                     << object_frame << " due to distance threshold.");
  }

  // Update the frame IDs based on distance from the static frame
  update_filter_frame_index(object_frame);
}

void ObjectMapTFServerROS::update_filter_frame_index(
    const std::string& object_frame) {
  auto it = filters_.find(object_frame);
  if (it == filters_.end() || it->second.empty()) {
    return;
  }

  // Calculate distances from base_link to each filter's position
  std::vector<std::pair<size_t, double>> filter_distances;
  filter_distances.reserve(it->second.size());

  const std::string base_link_frame = "taluy/base_link";

  for (size_t i = 0; i < it->second.size(); ++i) {
    const auto& transform = it->second[i]->getFilteredTransform();

    // Create a point in the static frame at the filter's position
    geometry_msgs::PointStamped point;
    point.header.frame_id = static_frame_;
    point.header.stamp = ros::Time(0);
    point.point.x = transform.transform.translation.x;
    point.point.y = transform.transform.translation.y;
    point.point.z = transform.transform.translation.z;

    // Try to transform point to base_link frame
    try {
      // Look up transform from static frame to base_link
      auto base_link_transform = tf_buffer_.lookupTransform(
          base_link_frame, static_frame_, ros::Time(0), ros::Duration(1.0));

      // Transform the point to base_link frame
      geometry_msgs::PointStamped point_in_base_link;
      tf2::doTransform(point, point_in_base_link, base_link_transform);

      // Calculate distance from base_link origin
      const double distance =
          std::sqrt(std::pow(point_in_base_link.point.x, 2) +
                    std::pow(point_in_base_link.point.y, 2) +
                    std::pow(point_in_base_link.point.z, 2));

      filter_distances.emplace_back(i, distance);
    } catch (tf2::TransformException& ex) {
      ROS_WARN_STREAM("Transform lookup failed: " << ex.what());
      // Fallback: use distance from static frame instead
      const double distance =
          std::sqrt(std::pow(transform.transform.translation.x, 2) +
                    std::pow(transform.transform.translation.y, 2) +
                    std::pow(transform.transform.translation.z, 2));
      filter_distances.emplace_back(i, distance);
    }
  }

  // Sort filters by distance (closest first)
  std::sort(filter_distances.begin(), filter_distances.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

  // Update frame IDs based on sorted distances
  for (size_t i = 0; i < filter_distances.size(); ++i) {
    auto& filter = it->second[filter_distances[i].first];
    auto transform = filter->getFilteredTransform();

    if (i == 0) {
      // Closest filter gets the base name without index
      transform.child_frame_id = object_frame;
    } else {
      // Other filters get indexed names (starting from 0)
      transform.child_frame_id = object_frame + "_" + std::to_string(i - 1);
    }

    // Update the filter with the new frame ID
    filter->updateFrameIndex(transform.child_frame_id);
  }
}

// Tf tree'de olmayan bir nesneyi tf tree'ye bağlıyoruz
std::optional<geometry_msgs::TransformStamped>
ObjectMapTFServerROS::transform_to_static_frame(
    const geometry_msgs::TransformStamped transform) {
  const auto parent_frame = transform.header.frame_id;
  const auto target_frame = transform.child_frame_id;
  const auto static_to_parent_transform =
      get_transform(static_frame_, parent_frame,
                    ros::Duration(4.0));  // parent'ı odoma göre tf alıyoruz

  if (!static_to_parent_transform.has_value()) {
    ROS_ERROR("Error occurred while looking up transform");
    return std::nullopt;
  }

  const auto parent_to_target_quaternion = tf2::Quaternion(
      transform.transform.rotation.x, transform.transform.rotation.y,
      transform.transform.rotation.z, transform.transform.rotation.w);

  auto parent_to_target_orientation =
      tf2::Matrix3x3{parent_to_target_quaternion};

  auto parent_to_target_translation = tf2::Vector3{
      transform.transform.translation.x, transform.transform.translation.y,
      transform.transform.translation.z};

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

  const auto static_to_target_translation =
      tf2::Vector3{static_to_parent_translation +
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
  static_transform.transform.rotation = tf2::toMsg(static_to_target_quaternion);

  return static_transform;
}

std::optional<geometry_msgs::TransformStamped>
ObjectMapTFServerROS::get_transform(const std::string& target_frame,
                                    const std::string& source_frame,
                                    const ros::Duration timeout) {
  try {
    auto transform = tf_buffer_.lookupTransform(target_frame, source_frame,
                                                ros::Time(0), timeout);
    return transform;
  } catch (tf2::TransformException& ex) {
    ROS_WARN_STREAM("Transform lookup failed: " << ex.what());
    return std::nullopt;
  }
}

}  // namespace auv_mapping

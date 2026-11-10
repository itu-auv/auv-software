/**
 * @file object_plane_fitter_node.cpp
 * @brief ROS 1 node for estimating object pose via RANSAC plane fitting
 *
 * This node subscribes to depth images, camera info, and YOLO detections,
 * then uses RANSAC to fit planes to detected objects and publishes their poses.
 */

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <map>
#include <mutex>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <string>
#include <vector>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "rclcpp/rclcpp.hpp"

class ObjectPlaneFitter {
 public:
  ObjectPlaneFitter(rclcpp::Node::SharedPtr nh)
      : nh_(nh),
        enabled_(false),
        camera_info_received_(false),
        original_camera_info_received_(false),
        tf_buffer_(nh->get_clock(), tf2::durationFromSec(10.0)),
        tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    // Load parameters
    // Use -1 to indicate "all classes" (no filtering)
    target_class_id_ = nh_->declare_parameter<int>("target_class_id", 4);
    ransac_distance_threshold_ =
        nh_->declare_parameter<double>("ransac_distance_threshold", 0.03);
    ransac_max_iterations_ =
        nh_->declare_parameter<int>("ransac_max_iterations", 1000);
    depth_scale_factor_ =
        nh_->declare_parameter<double>("depth_scale_factor", 1.0);
    min_inlier_ratio_ = nh_->declare_parameter<double>("min_inlier_ratio", 0.5);
    publish_debug_ = nh_->declare_parameter<bool>("publish_debug", true);
    output_frame_id_ =
        nh_->declare_parameter<std::string>("output_frame_id", "");
    sync_queue_size_ = nh_->declare_parameter<int>("sync_queue_size", 90);
    camera_optical_frame_ = nh_->declare_parameter<std::string>(
        "camera_optical_frame", "taluy/camera_depth_optical_frame");
    base_link_frame_ = nh_->declare_parameter<std::string>("base_link_frame",
                                                           "taluy/base_link");
    // Publishers
    object_transform_pub_ =
        nh_->create_publisher<geometry_msgs::msg::TransformStamped>(
            "object_transform_updates", 10);
    if (publish_debug_) {
      debug_cloud_pub_ = nh_->create_publisher<sensor_msgs::msg::PointCloud2>(
          "debug/cropped_cloud", 1);
      debug_inlier_cloud_pub_ =
          nh_->create_publisher<sensor_msgs::msg::PointCloud2>(
              "debug/inlier_cloud", 1);
    }

    // Enable service
    enable_service_ = nh_->create_service<std_srvs::srv::SetBool>(
        "~/enable",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
          enableCallback(*req, *res);
        });
    client_group_ = nh_->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    tracker_enable_client_ = nh_->create_client<std_srvs::srv::SetBool>(
        "tracker_enable", rclcpp::ServicesQoS(), client_group_);
    depth_enable_client_ = nh_->create_client<std_srvs::srv::SetBool>(
        "depth_enable", rclcpp::ServicesQoS(), client_group_);

    // Camera info subscriber (separate, latched)
    camera_info_sub_ = nh_->create_subscription<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::QoS(1).transient_local(),
        std::bind(&ObjectPlaneFitter::cameraInfoCallback, this,
                  std::placeholders::_1));
    original_camera_info_sub_ =
        nh_->create_subscription<sensor_msgs::msg::CameraInfo>(
            "original_camera_info", 1,
            std::bind(&ObjectPlaneFitter::originalCameraInfoCallback, this,
                      std::placeholders::_1));

    // Synchronized subscribers for depth and detections
    const auto qos = rclcpp::QoS(sync_queue_size_).get_rmw_qos_profile();
    depth_sub_.subscribe(nh_, "depth", qos);
    detection_sub_.subscribe(nh_, "detections", qos);

    // ApproximateTime synchronizer
    sync_.reset(
        new Sync(SyncPolicy(sync_queue_size_), depth_sub_, detection_sub_));
    sync_->registerCallback(std::bind(&ObjectPlaneFitter::syncCallback, this,
                                      std::placeholders::_1,
                                      std::placeholders::_2));

    RCLCPP_INFO(nh_->get_logger(), "[ObjectPlaneFitter] Node initialized");
    RCLCPP_INFO(nh_->get_logger(), "[ObjectPlaneFitter] Target class ID: %s",
                target_class_id_ < 0
                    ? "(all)"
                    : std::to_string(target_class_id_).c_str());
    RCLCPP_INFO(
        nh_->get_logger(),
        "[ObjectPlaneFitter] RANSAC threshold: %.4f m, max iterations: %d",
        ransac_distance_threshold_, ransac_max_iterations_);
    RCLCPP_INFO(nh_->get_logger(), "[ObjectPlaneFitter] Sync queue size: %d",
                sync_queue_size_);
    RCLCPP_INFO(
        nh_->get_logger(),
        "[ObjectPlaneFitter] BBox scaling enabled: YOLO boxes will be rescaled "
        "from original camera resolution to depth image resolution.");
  }

  bool enableCallback(std_srvs::srv::SetBool::Request& req,
                      std_srvs::srv::SetBool::Response& res) {
    if (req.data) {
      const bool depth_updated = setDepthEnabled(true);
      const bool tracker_updated = depth_updated && setTrackerEnabled(true);

      if (depth_updated && tracker_updated) {
        enabled_ = true;
        res.success = true;
        res.message = "Object plane fitter, TensorRT depth, and YOLO enabled";
      } else {
        enabled_ = false;
        setTrackerEnabled(false);
        setDepthEnabled(false);
        res.success = false;
        res.message =
            "Failed to enable TensorRT depth or YOLO; DA3 pipeline rolled back";
      }
    } else {
      enabled_ = false;
      const bool tracker_updated = setTrackerEnabled(false);
      const bool depth_updated = setDepthEnabled(false);
      res.success = tracker_updated && depth_updated;
      res.message =
          res.success ? "Object plane fitter, TensorRT depth, and YOLO disabled"
                      : "Object plane fitter disabled, but a dependency could "
                        "not be disabled";
    }

    if (res.success) {
      RCLCPP_INFO_STREAM(nh_->get_logger(), res.message);
    } else {
      RCLCPP_ERROR_STREAM(nh_->get_logger(), res.message);
    }
    return true;
  }

  bool setTrackerEnabled(bool enabled) {
    return setDependencyEnabled(tracker_enable_client_, "tracker_enable",
                                "YOLO tracker", enabled);
  }

  bool setDepthEnabled(bool enabled) {
    return setDependencyEnabled(depth_enable_client_, "depth_enable",
                                "TensorRT depth", enabled);
  }

  // Synchronous SetBool call; needs the client's callback group to be spun by
  // another executor thread than the caller's.
  bool callSetBool(
      const rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr& client,
      bool enabled, std_srvs::srv::SetBool::Response& response) {
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enabled;
    auto future = client->async_send_request(request);
    if (future.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
      return false;
    }
    response = *future.get();
    return true;
  }

  bool setDependencyEnabled(
      const rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr& client,
      const std::string& service_name, const std::string& dependency_name,
      bool enabled) {
    if (!client->service_is_ready()) {
      RCLCPP_ERROR_STREAM(nh_->get_logger(),
                          dependency_name << " enable service is unavailable: "
                                          << service_name);
      return false;
    }

    std_srvs::srv::SetBool::Response enable_response;
    if (!callSetBool(client, enabled, enable_response)) {
      RCLCPP_ERROR_STREAM(nh_->get_logger(),
                          "Failed to call "
                              << dependency_name
                              << " enable service: " << service_name);
      return false;
    }

    if (!enable_response.success) {
      RCLCPP_ERROR_STREAM(nh_->get_logger(),
                          dependency_name << " rejected enable=" << enabled
                                          << " via " << service_name << ": "
                                          << enable_response.message);
      return false;
    }

    return true;
  }

 private:
  void cameraInfoCallback(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_info_ = *msg;
    camera_info_received_ = true;
    RCLCPP_INFO_ONCE(nh_->get_logger(),
                     "[ObjectPlaneFitter] Scaled camera info received: %dx%d",
                     msg->width, msg->height);
  }

  void originalCameraInfoCallback(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg) {
    std::lock_guard<std::mutex> lock(original_camera_info_mutex_);
    original_camera_info_ = *msg;
    original_camera_info_received_ = true;
    RCLCPP_INFO_ONCE(nh_->get_logger(),
                     "[ObjectPlaneFitter] Original camera info received: %dx%d",
                     msg->width, msg->height);
  }

  void syncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg,
                    const vision_msgs::msg::Detection2DArray::ConstSharedPtr&
                        detections_msg) {
    if (!enabled_) {
      return;
    }
    // Check if both camera infos are available
    sensor_msgs::msg::CameraInfo cam_info;
    sensor_msgs::msg::CameraInfo orig_cam_info;
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      if (!camera_info_received_) {
        RCLCPP_WARN_THROTTLE(
            nh_->get_logger(), *nh_->get_clock(),
            static_cast<int>((5.0) * 1000),
            "[ObjectPlaneFitter] Waiting for scaled camera info...");
        return;
      }
      cam_info = camera_info_;
    }
    {
      std::lock_guard<std::mutex> lock(original_camera_info_mutex_);
      if (!original_camera_info_received_) {
        RCLCPP_WARN_THROTTLE(
            nh_->get_logger(), *nh_->get_clock(),
            static_cast<int>((5.0) * 1000),
            "[ObjectPlaneFitter] Waiting for original camera info...");
        return;
      }
      orig_cam_info = original_camera_info_;
    }

    // Convert depth image
    cv_bridge::CvImageConstPtr cv_depth;
    try {
      cv_depth = cv_bridge::toCvShare(depth_msg);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(nh_->get_logger(),
                   "[ObjectPlaneFitter] cv_bridge exception: %s", e.what());
      return;
    }

    // Get camera intrinsics
    double fx = cam_info.k[0];
    double fy = cam_info.k[4];
    double cx = cam_info.k[2];
    double cy = cam_info.k[5];

    if (fx == 0.0 || fy == 0.0) {
      RCLCPP_ERROR_THROTTLE(
          nh_->get_logger(), *nh_->get_clock(), static_cast<int>((5.0) * 1000),
          "[ObjectPlaneFitter] Invalid camera intrinsics (fx=%.2f, fy=%.2f)",
          fx, fy);
      return;
    }

    // Compute scale factors: YOLO boxes are in original resolution,
    // depth image is in resized resolution.
    int depth_w = cv_depth->image.cols;
    int depth_h = cv_depth->image.rows;
    int orig_w = static_cast<int>(orig_cam_info.width);
    int orig_h = static_cast<int>(orig_cam_info.height);

    double scale_x = 1.0;
    double scale_y = 1.0;
    if (orig_w > 0 && orig_h > 0) {
      scale_x = static_cast<double>(depth_w) / orig_w;
      scale_y = static_cast<double>(depth_h) / orig_h;
    }
    RCLCPP_DEBUG_THROTTLE(
        nh_->get_logger(), *nh_->get_clock(), static_cast<int>((5.0) * 1000),
        "[ObjectPlaneFitter] BBox scale: orig=%dx%d, depth=%dx%d, "
        "scale=(%.3f, %.3f)",
        orig_w, orig_h, depth_w, depth_h, scale_x, scale_y);

    // Process each detection
    for (const auto& detection : detections_msg->detections) {
      // Filter by class if specified (target_class_id_ < 0 means accept all)
      if (target_class_id_ >= 0) {
        bool class_match = false;
        for (const auto& result : detection.results) {
          if (std::stoi(result.hypothesis.class_id) == target_class_id_) {
            class_match = true;
            break;
          }
        }
        if (!class_match) {
          continue;
        }
      }

      // Get bounding box and scale from original to depth resolution
      int bbox_cx =
          static_cast<int>(detection.bbox.center.position.x * scale_x);
      int bbox_cy =
          static_cast<int>(detection.bbox.center.position.y * scale_y);
      int bbox_w = static_cast<int>(detection.bbox.size_x * scale_x);
      int bbox_h = static_cast<int>(detection.bbox.size_y * scale_y);

      int x_min = std::max(0, bbox_cx - bbox_w / 2);
      int y_min = std::max(0, bbox_cy - bbox_h / 2);
      int x_max = std::min(depth_w, bbox_cx + bbox_w / 2);
      int y_max = std::min(depth_h, bbox_cy + bbox_h / 2);

      if (x_max <= x_min || y_max <= y_min) {
        RCLCPP_WARN_THROTTLE(nh_->get_logger(), *nh_->get_clock(),
                             static_cast<int>((1.0) * 1000),
                             "[ObjectPlaneFitter] Invalid bounding box");
        continue;
      }

      // Extract 3D points from the ROI
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
          new pcl::PointCloud<pcl::PointXYZ>);

      for (int v = y_min; v < y_max; ++v) {
        for (int u = x_min; u < x_max; ++u) {
          float depth_value = getDepthValue(cv_depth->image, u, v);

          if (!std::isfinite(depth_value) || depth_value <= 0.0f) {
            continue;
          }

          // Apply depth scale
          float Z = depth_value * depth_scale_factor_;

          // Deproject to 3D
          float X = static_cast<float>((u - cx) * Z / fx);
          float Y = static_cast<float>((v - cy) * Z / fy);

          pcl::PointXYZ point;
          point.x = X;
          point.y = Y;
          point.z = Z;
          cloud->points.push_back(point);
        }
      }

      if (cloud->points.size() < 3) {
        RCLCPP_WARN_THROTTLE(
            nh_->get_logger(), *nh_->get_clock(),
            static_cast<int>((1.0) * 1000),
            "[ObjectPlaneFitter] Not enough valid points in ROI: %zu",
            cloud->points.size());
        continue;
      }

      cloud->width = cloud->points.size();
      cloud->height = 1;
      cloud->is_dense = false;

      // Publish debug cropped cloud
      if (publish_debug_ && debug_cloud_pub_->get_subscription_count() > 0) {
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header = depth_msg->header;
        debug_cloud_pub_->publish(cloud_msg);
      }

      // RANSAC plane fitting
      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

      pcl::SACSegmentation<pcl::PointXYZ> seg;
      seg.setOptimizeCoefficients(true);
      seg.setModelType(pcl::SACMODEL_PLANE);
      seg.setMethodType(pcl::SAC_RANSAC);
      seg.setDistanceThreshold(ransac_distance_threshold_);
      seg.setMaxIterations(ransac_max_iterations_);
      seg.setInputCloud(cloud);
      seg.segment(*inliers, *coefficients);

      if (inliers->indices.empty()) {
        RCLCPP_WARN_THROTTLE(
            nh_->get_logger(), *nh_->get_clock(),
            static_cast<int>((1.0) * 1000),
            "[ObjectPlaneFitter] RANSAC failed to find a plane");
        continue;
      }

      // Check inlier ratio
      double inlier_ratio =
          static_cast<double>(inliers->indices.size()) / cloud->points.size();
      if (inlier_ratio < min_inlier_ratio_) {
        RCLCPP_WARN_THROTTLE(
            nh_->get_logger(), *nh_->get_clock(),
            static_cast<int>((1.0) * 1000),
            "[ObjectPlaneFitter] Inlier ratio too low: %.2f < %.2f",
            inlier_ratio, min_inlier_ratio_);
        continue;
      }

      // Extract plane coefficients: ax + by + cz + d = 0
      float a = coefficients->values[0];
      float b = coefficients->values[1];
      float c = coefficients->values[2];

      // Compute centroid of inliers
      Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
      for (const auto& idx : inliers->indices) {
        centroid.x() += cloud->points[idx].x;
        centroid.y() += cloud->points[idx].y;
        centroid.z() += cloud->points[idx].z;
      }
      centroid /= static_cast<float>(inliers->indices.size());

      // Normal vector
      Eigen::Vector3f normal(a, b, c);
      normal.normalize();

      // Ensure normal points towards camera using dot product with centroid
      // vector (same logic as process_tracker_with_cloud) If
      // normal.dot(centroid) > 0, normal points away from camera, so flip it
      if (normal.dot(centroid) > 0) {
        normal = -normal;
      }

      // Compute quaternion from normal
      // Align Z-axis with the plane normal
      Eigen::Quaternionf quat = computeQuaternionFromNormal(normal);

      // Match naming from process_tracker_with_cloud
      std::string base_frame_id = "object";
      if (!detection.results.empty()) {
        int detection_id = std::stoi(detection.results[0].hypothesis.class_id);
        auto it = id_to_prop_name_.find(detection_id);
        if (it != id_to_prop_name_.end()) {
          base_frame_id = it->second;
        }
      }
      std::string frame_id = base_frame_id + "_closest";

      // Publish object transform (same pattern as process_tracker_with_cloud)
      geometry_msgs::msg::TransformStamped transform_msg;
      transform_msg.header = depth_msg->header;
      if (!output_frame_id_.empty()) {
        transform_msg.header.frame_id = output_frame_id_;
      }
      transform_msg.child_frame_id = frame_id;
      transform_msg.transform.translation.x = centroid.x();
      transform_msg.transform.translation.y = centroid.y();
      transform_msg.transform.translation.z = centroid.z();
      transform_msg.transform.rotation.x = quat.x();
      transform_msg.transform.rotation.y = quat.y();
      transform_msg.transform.rotation.z = quat.z();
      transform_msg.transform.rotation.w = quat.w();
      object_transform_pub_->publish(transform_msg);

      // Publish debug visualizations
      if (publish_debug_) {
        publishDebugVisualizations(depth_msg->header, inliers, cloud);
      }

      RCLCPP_DEBUG(nh_->get_logger(),
                   "[ObjectPlaneFitter] Plane fit: normal=(%.3f, %.3f, %.3f), "
                   "centroid=(%.3f, %.3f, %.3f), inliers=%zu (%.1f%%)",
                   normal.x(), normal.y(), normal.z(), centroid.x(),
                   centroid.y(), centroid.z(), inliers->indices.size(),
                   inlier_ratio * 100.0);
    }
  }

  float getDepthValue(const cv::Mat& depth_image, int u, int v) {
    switch (depth_image.type()) {
      case CV_32FC1:
        return depth_image.at<float>(v, u);
      case CV_16UC1:
        return static_cast<float>(depth_image.at<uint16_t>(v, u)) *
               0.001f;  // mm to m
      case CV_64FC1:
        return static_cast<float>(depth_image.at<double>(v, u));
      default:
        RCLCPP_ERROR_ONCE(
            nh_->get_logger(),
            "[ObjectPlaneFitter] Unsupported depth image type: %d",
            depth_image.type());
        return std::numeric_limits<float>::quiet_NaN();
    }
  }

  Eigen::Quaternionf computeQuaternionFromNormal(
      const Eigen::Vector3f& normal) {
    // Match process_tracker_with_cloud convention:
    // - Y-axis = plane normal (pointing toward camera)
    // - Z-axis = "up" direction (aligned with base_link Z via TF)
    // - X-axis = Y × Z (completes right-handed frame)

    Eigen::Vector3f y_axis = normal.normalized();

    // Choose reference vector for Z-axis (avoid parallel to normal)
    Eigen::Vector3f ref = (std::abs(y_axis.x()) < 0.9f)
                              ? Eigen::Vector3f::UnitX()
                              : Eigen::Vector3f::UnitY();

    // Z-axis is perpendicular to Y-axis, in the plane formed by ref and Y
    Eigen::Vector3f z_axis = ref.cross(y_axis).normalized();

    // X-axis completes the right-handed frame
    Eigen::Vector3f x_axis = y_axis.cross(z_axis).normalized();

    // Use TF to align Z-axis with robot's "up" direction (same as
    // process_tracker_with_cloud)
    try {
      geometry_msgs::msg::TransformStamped tf_msg = tf_buffer_.lookupTransform(
          camera_optical_frame_, base_link_frame_, tf2::TimePointZero,
          tf2::durationFromSec(0.05));
      tf2::Quaternion q(
          tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
          tf_msg.transform.rotation.z, tf_msg.transform.rotation.w);
      tf2::Matrix3x3 m(q);
      // Get the Z-axis of base_link in camera optical frame
      Eigen::Vector3f base_z_cam(m[0][2], m[1][2], m[2][2]);
      base_z_cam.normalize();
      // Flip Z-axis if it's pointing opposite to base_link's Z
      if (z_axis.dot(base_z_cam) < 0.0f) {
        z_axis = -z_axis;
        x_axis = -x_axis;
      }
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_STREAM_THROTTLE(
          nh_->get_logger(), *nh_->get_clock(), static_cast<int>((5.0) * 1000),
          "[ObjectPlaneFitter] TF lookup failed: " << ex.what()
                                                   << ". Using fallback.");
      // Fallback: assume "up" is negative Y in camera optical frame
      if (z_axis.y() > 0) {
        z_axis = -z_axis;
        x_axis = -x_axis;
      }
    }

    // Build rotation matrix
    Eigen::Matrix3f rot;
    rot.col(0) = x_axis;
    rot.col(1) = y_axis;
    rot.col(2) = z_axis;

    // Ensure proper rotation matrix (det = +1)
    if (rot.determinant() < 0.0f) {
      rot.col(2) *= -1.0f;
    }

    return Eigen::Quaternionf(rot);
  }

  void publishDebugVisualizations(
      const std_msgs::msg::Header& header,
      const pcl::PointIndices::Ptr& inliers,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Publish inlier point cloud
    if (debug_inlier_cloud_pub_->get_subscription_count() > 0) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(cloud);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);

      sensor_msgs::msg::PointCloud2 inlier_cloud_msg;
      pcl::toROSMsg(*inlier_cloud, inlier_cloud_msg);
      inlier_cloud_msg.header = header;
      debug_inlier_cloud_pub_->publish(inlier_cloud_msg);
    }
  }

  // ROS handles
  rclcpp::Node::SharedPtr nh_;
  rclcpp::CallbackGroup::SharedPtr client_group_;

  // Publishers
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
      object_transform_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
      debug_inlier_cloud_pub_;

  // Service Servers
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr tracker_enable_client_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr depth_enable_client_;
  bool enabled_;

  // Subscribers
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      original_camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
  message_filters::Subscriber<vision_msgs::msg::Detection2DArray>
      detection_sub_;

  // Synchronization
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::Image, vision_msgs::msg::Detection2DArray>
      SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  std::shared_ptr<Sync> sync_;

  // Camera info
  sensor_msgs::msg::CameraInfo camera_info_;
  bool camera_info_received_;
  std::mutex camera_info_mutex_;

  sensor_msgs::msg::CameraInfo original_camera_info_;
  bool original_camera_info_received_;
  std::mutex original_camera_info_mutex_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string camera_optical_frame_;
  std::string base_link_frame_;

  // Parameters
  int target_class_id_;  // -1 means accept all classes
  double ransac_distance_threshold_;
  int ransac_max_iterations_;
  double depth_scale_factor_;
  double min_inlier_ratio_;
  bool publish_debug_;
  std::string output_frame_id_;
  int sync_queue_size_;
  std::map<int, std::string> id_to_prop_name_ = {
      {0, "gate_sawfish_link"}, {1, "gate_shark_link"},
      {2, "red_pipe_link"},     {3, "white_pipe_link"},
      {4, "torpedo_map_link"},  {5, "torpedo_hole_link"},
      {6, "bin_whole_link"},    {7, "octagon_link"}};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto nh = std::make_shared<rclcpp::Node>("object_plane_fitter");
  ObjectPlaneFitter node(nh);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(nh);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}

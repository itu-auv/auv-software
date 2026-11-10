/**
 * auv_mapping
 * ROS node for processing PointCloud2 data with YOLO detections
 * Inspired by ultralytics_ros tracker_with_cloud_node
 */

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <opencv2/core/core.hpp>

// PCL Libraries
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>

class ProcessTrackerWithCloud {
 private:
  // ROS handlers
  rclcpp::Node::SharedPtr nh_;
  rclcpp::CallbackGroup::SharedPtr client_group_;

  // Parameters
  std::string camera_info_topic_, lidar_topic_, yolo_result_topic_;
  std::string yolo_3d_result_topic_;
  float cluster_tolerance_, voxel_leaf_size_;
  int min_cluster_size_, max_cluster_size_;
  float roi_expansion_factor_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr detection_cloud_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr
      object_transform_pub_;

  // Service Servers
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr tracker_enable_client_;
  bool enabled_;
  std::string tracker_enable_service_;

  // Subscribers and synchronizers
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
  message_filters::Subscriber<vision_msgs::msg::Detection2DArray> yolo_result_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::msg::CameraInfo, sensor_msgs::msg::PointCloud2,
      vision_msgs::msg::Detection2DArray>
      SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  std::shared_ptr<Sync> sync_;

  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string camera_optical_frame_;
  std::string base_link_frame_;

  // Camera model
  sensor_msgs::msg::CameraInfo cam_info_;

  // Last processing time
  rclcpp::Time last_call_time_{0, 0, RCL_ROS_TIME};

  // Set to store detection IDs to skip
  std::set<int> skip_detection_ids;

  // Mapping from detection ID to prop name
  std::map<int, std::string> id_to_prop_name = {
      {0, "gate_sawfish_link"}, {1, "gate_shark_link"},
      {2, "red_pipe_link"},     {3, "white_pipe_link"},
      {4, "torpedo_map_link"},  {5, "torpedo_hole_link"},
      {6, "bin_whole_link"},    {7, "octagon_link"}};

 public:
  ProcessTrackerWithCloud(rclcpp::Node::SharedPtr nh)
      : nh_(nh),
        enabled_(false),
        tf_buffer_(nh->get_clock(), tf2::durationFromSec(10.0)),
        tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    // Load parameters
    camera_info_topic_ = nh_->declare_parameter<std::string>(
        "camera_info_topic", "camera_info");
    lidar_topic_ =
        nh_->declare_parameter<std::string>("lidar_topic", "points_raw");
    yolo_result_topic_ = nh_->declare_parameter<std::string>(
        "yolo_result_topic", "yolo_result");
    cluster_tolerance_ = nh_->declare_parameter<double>("cluster_tolerance", 0.3);
    voxel_leaf_size_ = nh_->declare_parameter<double>("voxel_leaf_size", 0.1);
    min_cluster_size_ = nh_->declare_parameter<int>("min_cluster_size", 100);
    max_cluster_size_ = nh_->declare_parameter<int>("max_cluster_size", 10000);
    roi_expansion_factor_ = nh_->declare_parameter<double>(
        "roi_expansion_factor", 1.1);  // 10% expansion
    tracker_enable_service_ = nh_->declare_parameter<std::string>(
        "tracker_enable_service", "/tracker_node_realsense/enable");

    std::vector<int64_t> skip_ids = nh_->declare_parameter<std::vector<int64_t>>(
        "skip_detection_ids", std::vector<int64_t>{});
    if (skip_ids.empty()) {
      // Set default values
      skip_detection_ids = {0, 1, 2, 3, 5, 6, 7};  // torpedo_hole_link
    } else {
      // Use parameter values
      skip_detection_ids.clear();
      skip_detection_ids.insert(skip_ids.begin(), skip_ids.end());
    }

    // Initialize publishers
    detection_cloud_pub_ = nh_->create_publisher<sensor_msgs::msg::PointCloud2>(
        "detection_cloud", 1);
    object_transform_pub_ =
        nh_->create_publisher<geometry_msgs::msg::TransformStamped>(
            "update_object_transforms", 10);

    // Initialize service server
    enable_service_ = nh_->create_service<std_srvs::srv::SetBool>(
        "~/enable",
        [this](const std::shared_ptr<std_srvs::srv::SetBool::Request> req,
               std::shared_ptr<std_srvs::srv::SetBool::Response> res) {
          enableCallback(*req, *res);
        });
    client_group_ = nh_->create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);
    if (!tracker_enable_service_.empty()) {
      tracker_enable_client_ = nh_->create_client<std_srvs::srv::SetBool>(
          tracker_enable_service_, rclcpp::ServicesQoS(), client_group_);
    }

    // Initialize subscribers
    const auto qos = rclcpp::QoS(10).get_rmw_qos_profile();
    camera_info_sub_.subscribe(nh_, camera_info_topic_, qos);
    lidar_sub_.subscribe(nh_, lidar_topic_, qos);
    yolo_result_sub_.subscribe(nh_, yolo_result_topic_, qos);

    // Configure synchronization
    sync_ = std::make_shared<Sync>(SyncPolicy(10), camera_info_sub_, lidar_sub_,
                                   yolo_result_sub_);
    sync_->registerCallback(std::bind(
        &ProcessTrackerWithCloud::syncCallback, this, std::placeholders::_1,
        std::placeholders::_2, std::placeholders::_3));

    // Create TF broadcaster
    tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster(nh_));
    camera_optical_frame_ = nh_->declare_parameter<std::string>(
        "camera_optical_frame", "taluy/camera_depth_optical_frame");
    base_link_frame_ = nh_->declare_parameter<std::string>("base_link_frame",
                                                           "taluy/base_link");
  }

  // Synchronous SetBool call; needs the client's callback group to be spun by
  // another executor thread than the caller's.
  bool callSetBool(const rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr& client,
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

  bool enableCallback(std_srvs::srv::SetBool::Request& req,
                      std_srvs::srv::SetBool::Response& res) {
    enabled_ = req.data;
    bool tracker_updated = setTrackerEnabled(enabled_);
    res.success = true;
    res.message = "Successfully set enabled  to " + std::to_string(enabled_);
    if (!tracker_updated) {
      res.message += "; YOLO tracker state was not updated";
    }
    RCLCPP_INFO_STREAM(nh_->get_logger(), res.message);
    return true;
  }

  bool setTrackerEnabled(bool enabled) {
    if (tracker_enable_service_.empty()) {
      return true;
    }

    if (!tracker_enable_client_->service_is_ready()) {
      RCLCPP_WARN_STREAM(nh_->get_logger(), "Could not set YOLO tracker via "
                      << tracker_enable_service_
                      << ": service is not available");
      return false;
    }

    std_srvs::srv::SetBool::Response tracker_enable_response;
    if (!callSetBool(tracker_enable_client_, enabled, tracker_enable_response)) {
      RCLCPP_WARN_STREAM(nh_->get_logger(), "Failed to call YOLO tracker enable service "
                      << tracker_enable_service_);
      return false;
    }

    if (!tracker_enable_response.success) {
      RCLCPP_WARN_STREAM(nh_->get_logger(), "YOLO tracker rejected enable="
                      << enabled << ": "
                      << tracker_enable_response.message);
      return false;
    }

    return true;
  }

  // Callback for synchronized messages
  // Callback for synchronized messages
  void syncCallback(
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info_msg,
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr& cloud_msg,
      const vision_msgs::msg::Detection2DArray::ConstSharedPtr& yolo_result_msg) {
    if (!enabled_) {
      return;
    }

    // Update camera model
    cam_info_ = *camera_info_msg;

    // Record call time
    rclcpp::Time current_call_time = nh_->now();
    rclcpp::Duration callback_interval = current_call_time - last_call_time_;
    last_call_time_ = current_call_time;

    // Skip if no YOLO detections
    if (yolo_result_msg->detections.empty()) {
      return;
    }

    // Convert point cloud to PCL format
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Skip if point cloud is empty
    if (cloud->points.empty()) {
      return;
    }

    // Downsample point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud =
        downsampleCloud(cloud);

    // Prepare data structures for 3D detections
    vision_msgs::msg::Detection3DArray detections3d_msg;
    sensor_msgs::msg::PointCloud2 detection_cloud_msg;
    visualization_msgs::msg::MarkerArray object_markers, plane_markers;

    // Set header information
    detections3d_msg.header = cloud_msg->header;
    detections3d_msg.header.stamp = yolo_result_msg->header.stamp;

    // Store combined point cloud for all detections
    pcl::PointCloud<pcl::PointXYZ> combined_detection_cloud;

    // Counter for processed detections
    int processed_detection_count = 0;

    // Process each YOLO detection
    for (size_t i = 0; i < yolo_result_msg->detections.size(); i++) {
      const auto& detection = yolo_result_msg->detections[i];

      // Check detection ID, skip if it's in the skip list
      if (!detection.results.empty()) {
        int detection_id = std::stoi(detection.results[0].hypothesis.class_id);
        if (skip_detection_ids.find(detection_id) != skip_detection_ids.end()) {
          continue;  // Skip this detection
        }
      }

      // Apply ROI filter and get detection points
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);

      // Use bounding box
      processPointsWithBbox(downsampled_cloud, detection, detection_cloud);

      if (detection_cloud->points.empty()) {
        continue;
      }

      // Determine detection ID and prop name here, after bbox filtering
      int detection_id = -1;
      std::string base_frame_id = "object";  // Default name

      // If detection results exist, use the detection class ID directly
      if (!detection.results.empty()) {
        // Get the first (or only) result of the detection
        detection_id = std::stoi(detection.results[0].hypothesis.class_id);

        // Convert ID to prop name
        if (detection_id >= 0 &&
            id_to_prop_name.find(detection_id) != id_to_prop_name.end()) {
          base_frame_id = id_to_prop_name[detection_id];
        }
      }

      // Euclidean cluster extraction uygula
      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters =
          euclideanClusterExtraction(detection_cloud);

      // Skip to next detection if no clusters found
      if (clusters.empty()) {
        continue;
      }

      // Print number of clusters found
      RCLCPP_INFO(nh_->get_logger(), "For detection ID %d (%s) found %zu clusters", detection_id,
               (detection_id >= 0 && id_to_prop_name.find(detection_id) !=
                                         id_to_prop_name.end()
                    ? id_to_prop_name[detection_id].c_str()
                    : "unknown"),
               clusters.size());

      // Variables to find the closest cluster
      size_t closest_cluster_idx = 0;
      float min_squared_distance = std::numeric_limits<float>::max();
      std::vector<Eigen::Vector4f> centroids(clusters.size());
      std::vector<Eigen::Matrix3f> rotation_matrices(clusters.size());
      std::vector<bool> success_flags(clusters.size(), false);

      // Process plane segmentation and PCA for each cluster, calculate centers
      for (size_t cluster_idx = 0; cluster_idx < clusters.size();
           cluster_idx++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_cloud =
            clusters[cluster_idx];

        // Skip if cluster is empty
        if (cluster_cloud->points.empty()) {
          continue;
        }

        // Apply plane segmentation and surface transformation (PCA)
        success_flags[cluster_idx] =
            planeSegmentationAndPCA(cluster_cloud, centroids[cluster_idx],
                                    rotation_matrices[cluster_idx], tf_buffer_,
                                    camera_optical_frame_, base_link_frame_);

        // Find closest cluster (smallest centroid norm squared)
        if (success_flags[cluster_idx]) {
          float squared_distance =
              centroids[cluster_idx][0] * centroids[cluster_idx][0] +
              centroids[cluster_idx][1] * centroids[cluster_idx][1] +
              centroids[cluster_idx][2] * centroids[cluster_idx][2];

          if (squared_distance < min_squared_distance) {
            min_squared_distance = squared_distance;
            closest_cluster_idx = cluster_idx;
          }
        }
      }

      RCLCPP_INFO(nh_->get_logger(), "Closest cluster index: %zu with squared distance: %f",
               closest_cluster_idx, min_squared_distance);

      // Now process all clusters and create detections
      for (size_t cluster_idx = 0; cluster_idx < clusters.size();
           cluster_idx++) {
        // Skip if cluster is empty or PCA failed
        if (!success_flags[cluster_idx]) {
          continue;
        }

        // Create unique frame name for each cluster
        const std::string base_name = base_frame_id;
        std::string frame_id =
            (cluster_idx == closest_cluster_idx)
                ? base_name + "_closest"
                : base_name + "_cluster_" + std::to_string(cluster_idx);

        // Create 3D detection message and marker using calculated centroid and
        // rotation matrix
        createAndPublishDetection(
            detections3d_msg, object_markers, plane_markers,
            clusters[cluster_idx], centroids[cluster_idx],
            rotation_matrices[cluster_idx], detection.results,
            cloud_msg->header, callback_interval.seconds(), frame_id);
        processed_detection_count++;

        // Combine detection points
        combined_detection_cloud += *(clusters[cluster_idx]);
      }
    }

    // Check processed detection count
    if (processed_detection_count == 0) {
      return;
    }

    // Convert combined point cloud to ROS message
    pcl::toROSMsg(combined_detection_cloud, detection_cloud_msg);
    detection_cloud_msg.header = cloud_msg->header;

    // Publish processed data
    detection_cloud_pub_->publish(detection_cloud_msg);
  }

  // Process point cloud using 2D bounding box
  void processPointsWithBbox(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const vision_msgs::msg::Detection2D& detection,
      pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud) {
    try {
      int points_in_bbox = 0;

      // Expand detection box using roi_expansion_factor parameter
      float min_x = detection.bbox.center.position.x -
                    (detection.bbox.size_x / 2) * roi_expansion_factor_;
      float max_x = detection.bbox.center.position.x +
                    (detection.bbox.size_x / 2) * roi_expansion_factor_;
      float min_y = detection.bbox.center.position.y -
                    (detection.bbox.size_y / 2) * roi_expansion_factor_;
      float max_y = detection.bbox.center.position.y +
                    (detection.bbox.size_y / 2) * roi_expansion_factor_;

      // Check each point in the point cloud
      for (const auto& point : cloud->points) {
        // NaN check
        if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
          continue;
        }

        // Manual projection calculation
        if (point.z <= 0) {
          continue;  // Skip points with negative or zero Z values
        }

        // Get camera parameters
        const double fx = cam_info_.k[0];  // Focal length in x
        const double fy = cam_info_.k[4];  // Focal length in y
        const double cx = cam_info_.k[2];  // Center x
        const double cy = cam_info_.k[5];  // Center y

        // Manual projection calculation
        double inv_z = 1.0 / point.z;
        cv::Point2d uv;
        uv.x = fx * point.x * inv_z + cx;
        uv.y = fy * point.y * inv_z + cy;

        // Check projection values
        if (std::isnan(uv.x) || std::isnan(uv.y)) {
          continue;
        }

        RCLCPP_DEBUG(nh_->get_logger(), "Point processed successfully");

        // Check if point is within ROI
        if (point.z > 0 && uv.x >= min_x && uv.x <= max_x && uv.y >= min_y &&
            uv.y <= max_y) {
          detection_cloud->points.push_back(point);
          points_in_bbox++;
        }
      }

      if (points_in_bbox < 20) {
        RCLCPP_DEBUG(nh_->get_logger(), "Few points (%d) found in bounding box", points_in_bbox);
      }

    } catch (const std::exception& e) {
      RCLCPP_ERROR_STREAM(nh_->get_logger(), "Exception in processPointsWithBbox: " << e.what());
      detection_cloud->points.clear();
    }
  }

  // Euclidean Cluster Extraction - returns all clusters
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> euclideanClusterExtraction(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

    // Skip clustering if too few points and return existing points as single
    // cluster
    if (cloud->points.size() < min_cluster_size_ || cloud->points.size() < 20) {
      clusters.push_back(cloud);
      return clusters;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    // Set clustering parameters
    ec.setClusterTolerance(
        cluster_tolerance_);  // Maximum distance between points
    // Use dynamic min_cluster_size - should contain at least half of available
    // points
    ec.setMinClusterSize(std::min(min_cluster_size_,
                                  static_cast<int>(cloud->points.size() / 2)));
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // If no clusters found, return all points as single cluster
    if (cluster_indices.empty()) {
      clusters.push_back(cloud);
      return clusters;
    }

    // Create and return all clusters
    for (const auto& indices : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
          new pcl::PointCloud<pcl::PointXYZ>);

      // Get points in cluster
      for (const auto& idx : indices.indices) {
        cloud_cluster->push_back((*cloud)[idx]);
      }

      // Calculate cluster center (for information)
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud_cluster, centroid);
      float distance = centroid.norm();  // Distance from center

      // Add cluster to list
      clusters.push_back(cloud_cluster);
    }

    return clusters;
  }

  // -----------------------------------------------------------------------------
  //  Plane segmentation + PCA
  // -----------------------------------------------------------------------------
  bool planeSegmentationAndPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                               Eigen::Vector4f& centroid,
                               Eigen::Matrix3f& rotation_matrix,
                               const tf2_ros::Buffer& tf_buffer,
                               const std::string& camera_optical_frame_,
                               const std::string& base_link_frame_) {
    /* ------------------------------------------------------------------------*/
    /* 0) INITIAL CHECKS */
    /* ------------------------------------------------------------------------*/
    if (cloud->empty()) return false;

    /* ------------------------------------------------------------------------*/
    /* 1) CENTROID */
    /* ------------------------------------------------------------------------*/
    pcl::compute3DCentroid(*cloud, centroid);

    if (cloud->size() < 10) {  // few points → identity
      rotation_matrix.setIdentity();
      return true;
    }

    /* ------------------------------------------------------------------------*/
    /* 2) PCA (backup) */
    /* ------------------------------------------------------------------------*/
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    const Eigen::Matrix3f pca_eig = pca.getEigenVectors();

    /* ------------------------------------------------------------------------*/
    /* 3) RANSAC for plane */
    /* ------------------------------------------------------------------------*/
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);  // 1 cm
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.size() < 5) {  // Failed → use PCA
      rotation_matrix = pca_eig;
      return true;
    }

    /* ------------------------------------------------------------------------*/
    /* 4) Y-axis looking from camera (plane normal) */
    Eigen::Vector3f n(coeff->values[0], coeff->values[1], coeff->values[2]);
    n.normalize();

    Eigen::Vector3f c_vec(centroid[0], centroid[1], centroid[2]);
    const Eigen::Vector3f y_axis = (-n.dot(c_vec) < n.dot(c_vec)) ? -n : n;

    /* ------------------------------------------------------------------------*/
    /* 5) Selection of Z-axis */
    Eigen::Vector3f z_axis, ref = (std::abs(y_axis.x()) < 0.9f)
                                      ? Eigen::Vector3f::UnitX()
                                      : Eigen::Vector3f::UnitY();
    z_axis = ref.cross(y_axis).normalized();
    Eigen::Vector3f x_axis = y_axis.cross(z_axis).normalized();

    try {
      geometry_msgs::msg::TransformStamped tf_msg =
          tf_buffer.lookupTransform(camera_optical_frame_, base_link_frame_,
                                    tf2::TimePointZero,
                                    tf2::durationFromSec(0.05));
      tf2::Quaternion q(
          tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
          tf_msg.transform.rotation.z, tf_msg.transform.rotation.w);
      tf2::Matrix3x3 m(q);
      Eigen::Vector3f base_z_cam(m[0][2], m[1][2], m[2][2]);
      base_z_cam.normalize();
      if (z_axis.dot(base_z_cam) < 0.0f) {
        z_axis = -z_axis;
        x_axis = -x_axis;
      }
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_STREAM_THROTTLE(nh_->get_logger(), *nh_->get_clock(), static_cast<int>((5.0) * 1000), "TF lookup failed: " << ex.what());
    }

    rotation_matrix.col(0) = x_axis;
    rotation_matrix.col(1) = y_axis;
    rotation_matrix.col(2) = z_axis;
    if (rotation_matrix.determinant() < 0.0f) rotation_matrix.col(2) *= -1.0f;
    return true;
  }

  // Create and publish 3D detection message and marker
  void createAndPublishDetection(
      vision_msgs::msg::Detection3DArray& detections3d_msg,
      visualization_msgs::msg::MarkerArray& object_markers,
      visualization_msgs::msg::MarkerArray& plane_markers,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const Eigen::Vector4f& centroid, const Eigen::Matrix3f& rotation_matrix,
      const std::vector<vision_msgs::msg::ObjectHypothesisWithPose>& results,
      const std_msgs::msg::Header& header, const double& duration,
      const std::string& frame_id) {
    if (cloud->points.empty()) {
      return;
    }
    Eigen::Matrix3f rot = rotation_matrix;  // Create copy
    Eigen::Quaternionf q(rot);
    // Create TransformStamped message and publish to object_transform_updates
    // topic
    geometry_msgs::msg::TransformStamped transform_msg;
    transform_msg.header.stamp = header.stamp;
    transform_msg.header.frame_id = camera_optical_frame_;  // Point cloud frame
    transform_msg.child_frame_id = frame_id;                // Cluster/object ID

    // Set position information - use as calculated
    transform_msg.transform.translation.x = centroid[0];
    transform_msg.transform.translation.y = centroid[1];
    transform_msg.transform.translation.z = centroid[2];

    // Set orientation information
    transform_msg.transform.rotation.x = q.x();
    transform_msg.transform.rotation.y = q.y();
    transform_msg.transform.rotation.z = q.z();
    transform_msg.transform.rotation.w = q.w();

    // Publish TransformStamped message
    object_transform_pub_->publish(transform_msg);
  }
  // Downsample PointCloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // Skip transformation if point count is too low
    if (cloud->points.size() < 100) {
      return cloud;
    }

    // Create Voxel Grid
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_,
                           voxel_leaf_size_);  // in meters
    voxel_grid.filter(*downsampled_cloud);

    return downsampled_cloud;
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto nh = std::make_shared<rclcpp::Node>("process_tracker_with_cloud");
  ProcessTrackerWithCloud node(nh);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(nh);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}

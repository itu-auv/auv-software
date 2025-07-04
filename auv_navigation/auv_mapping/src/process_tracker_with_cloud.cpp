/**
 * auv_mapping
 * ROS node for processing PointCloud2 data with YOLO detections
 * Inspired by ultralytics_ros tracker_with_cloud_node
 */

#include <geometry_msgs/TransformStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <ultralytics_ros/YoloResult.h>
#include <vision_msgs/Detection3DArray.h>
#include <visualization_msgs/MarkerArray.h>

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
#include <pcl_ros/transforms.h>

class ProcessTrackerWithCloud {
 private:
  // ROS handlers
  ros::NodeHandle nh_, pnh_;

  // Parameters
  std::string camera_info_topic_, lidar_topic_, yolo_result_topic_;
  std::string yolo_3d_result_topic_;
  float cluster_tolerance_, voxel_leaf_size_;
  int min_cluster_size_, max_cluster_size_;
  float roi_expansion_factor_;

  // Publishers
  ros::Publisher detection_cloud_pub_;
  ros::Publisher object_transform_pub_;

  // Subscribers and synchronizers
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub_;
  message_filters::Subscriber<ultralytics_ros::YoloResult> yolo_result_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::PointCloud2,
      ultralytics_ros::YoloResult>
      SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  boost::shared_ptr<Sync> sync_;

  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string camera_optical_frame_;
  std::string base_link_frame_;

  // Camera model
  image_geometry::PinholeCameraModel cam_model_;

  // Last processing time
  ros::Time last_call_time_;

  // Set to store detection IDs to skip
  std::set<int> skip_detection_ids;

  // Mapping from detection ID to prop name
  std::map<int, std::string> id_to_prop_name = {
      {8, "red_buoy_link"},       {7, "path_link"},
      {9, "bin_whole_link"},      {12, "torpedo_map_link"},
      {13, "torpedo_hole_link"},  {1, "gate_left_link"},
      {2, "gate_right_link"},     {3, "gate_blue_arrow_link"},
      {4, "gate_red_arrow_link"}, {5, "gate_middle_part_link"},
      {14, "octagon_link"}};

 public:
  ProcessTrackerWithCloud()
      : pnh_("~"),
        tf_buffer_(ros::Duration(10.0)),
        tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    // Load parameters
    pnh_.param<std::string>("camera_info_topic", camera_info_topic_,
                            "camera_info");
    pnh_.param<std::string>("lidar_topic", lidar_topic_, "points_raw");
    pnh_.param<std::string>("yolo_result_topic", yolo_result_topic_,
                            "yolo_result");
    pnh_.param<float>("cluster_tolerance", cluster_tolerance_, 0.3);
    pnh_.param<float>("voxel_leaf_size", voxel_leaf_size_, 0.1);
    pnh_.param<int>("min_cluster_size", min_cluster_size_, 100);
    pnh_.param<int>("max_cluster_size", max_cluster_size_, 10000);
    pnh_.param<float>("roi_expansion_factor", roi_expansion_factor_,
                      1.1);  // 10% expansion

    // Get detection IDs to skip as parameter (default: skip IDs 7 and 13)
    std::vector<int> skip_ids;
    pnh_.getParam("skip_detection_ids", skip_ids);
    if (skip_ids.empty()) {
      // Set default values
      skip_detection_ids = {7, 13};  // path_link and torpedo_hole_link
    } else {
      // Use parameter values
      skip_detection_ids.clear();
      skip_detection_ids.insert(skip_ids.begin(), skip_ids.end());
    }

    // Initialize publishers
    detection_cloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>("detection_cloud", 1);
    object_transform_pub_ = nh_.advertise<geometry_msgs::TransformStamped>(
        "update_object_transforms", 10);

    // Initialize subscribers
    camera_info_sub_.subscribe(nh_, camera_info_topic_, 10);
    lidar_sub_.subscribe(nh_, lidar_topic_, 10);
    yolo_result_sub_.subscribe(nh_, yolo_result_topic_, 10);

    // Configure synchronization
    sync_ = boost::make_shared<Sync>(SyncPolicy(10), camera_info_sub_,
                                     lidar_sub_, yolo_result_sub_);
    sync_->registerCallback(
        boost::bind(&ProcessTrackerWithCloud::syncCallback, this, _1, _2, _3));

    // Create TF broadcaster
    tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());
    pnh_.param<std::string>("camera_optical_frame", camera_optical_frame_,
                            "taluy/camera_depth_optical_frame");
    base_link_frame_ = "taluy/base_link";
  }

  // Callback for synchronized messages
  void syncCallback(
      const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
      const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
      const ultralytics_ros::YoloResultConstPtr& yolo_result_msg) {
    // Update camera model
    cam_model_.fromCameraInfo(camera_info_msg);

    // Record call time
    ros::Time current_call_time = ros::Time::now();
    ros::Duration callback_interval = current_call_time - last_call_time_;
    last_call_time_ = current_call_time;

    // Skip if no YOLO detections
    if (yolo_result_msg->detections.detections.empty()) {
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
    vision_msgs::Detection3DArray detections3d_msg;
    sensor_msgs::PointCloud2 detection_cloud_msg;
    visualization_msgs::MarkerArray object_markers, plane_markers;

    // Set header information
    detections3d_msg.header = cloud_msg->header;
    detections3d_msg.header.stamp = yolo_result_msg->header.stamp;

    // Store combined point cloud for all detections
    pcl::PointCloud<pcl::PointXYZ> combined_detection_cloud;

    // Counter for processed detections
    int processed_detection_count = 0;

    // Process each YOLO detection
    for (size_t i = 0; i < yolo_result_msg->detections.detections.size(); i++) {
      const auto& detection = yolo_result_msg->detections.detections[i];

      // Check detection ID, skip if it's in the skip list
      if (!detection.results.empty()) {
        int detection_id = detection.results[0].id;
        if (skip_detection_ids.find(detection_id) != skip_detection_ids.end()) {
          continue;  // Skip this detection
        }
      }

      // Apply ROI filter and get detection points
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);

      if (yolo_result_msg->masks.empty()) {
        // Use bounding box if no mask available
        processPointsWithBbox(downsampled_cloud, detection, detection_cloud);
      } else {
        // Use mask if available
        // processPointsWithMask(cloud, yolo_result_msg->masks[i],
        // detection_cloud); NOTE: Skipping mask processing for now
        continue;
      }

      if (detection_cloud->points.empty()) {
        continue;
      }

      // Determine detection ID and prop name here, after bbox filtering
      int detection_id = -1;
      std::string base_frame_id = "object";  // Default name

      // If detection results exist, use the detection class ID directly
      if (!detection.results.empty()) {
        // Get the first (or only) result of the detection
        detection_id = detection.results[0].id;

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
      ROS_INFO("For detection ID %d (%s) found %zu clusters", detection_id,
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

      ROS_INFO("Closest cluster index: %zu with squared distance: %f",
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
            cloud_msg->header, callback_interval.toSec(), frame_id);
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
    detection_cloud_pub_.publish(detection_cloud_msg);
  }

  // Process point cloud using 2D bounding box
  void processPointsWithBbox(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const vision_msgs::Detection2D& detection,
      pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud) {
    try {
      int points_in_bbox = 0;

      // Expand detection box using roi_expansion_factor parameter
      float min_x = detection.bbox.center.x -
                    (detection.bbox.size_x / 2) * roi_expansion_factor_;
      float max_x = detection.bbox.center.x +
                    (detection.bbox.size_x / 2) * roi_expansion_factor_;
      float min_y = detection.bbox.center.y -
                    (detection.bbox.size_y / 2) * roi_expansion_factor_;
      float max_y = detection.bbox.center.y +
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
        const double fx = cam_model_.fx();  // Focal length x
        const double fy = cam_model_.fy();  // Focal length y
        const double cx = cam_model_.cx();  // Optical center x
        const double cy = cam_model_.cy();  // Optical center y

        // Manual projection calculation
        double inv_z = 1.0 / point.z;
        cv::Point2d uv;
        uv.x = fx * point.x * inv_z + cx;
        uv.y = fy * point.y * inv_z + cy;

        // Check projection values
        if (std::isnan(uv.x) || std::isnan(uv.y)) {
          continue;
        }

        ROS_DEBUG("Point processed successfully");

        // Check if point is within ROI
        if (point.z > 0 && uv.x >= min_x && uv.x <= max_x && uv.y >= min_y &&
            uv.y <= max_y) {
          detection_cloud->points.push_back(point);
          points_in_bbox++;
        }
      }

      if (points_in_bbox < 20) {
        ROS_DEBUG("Few points (%d) found in bounding box", points_in_bbox);
      }

    } catch (const std::exception& e) {
      ROS_ERROR_STREAM("Exception in processPointsWithBbox: " << e.what());
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
      geometry_msgs::TransformStamped tf_msg =
          tf_buffer.lookupTransform(camera_optical_frame_, base_link_frame_,
                                    ros::Time(0), ros::Duration(0.05));
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
      ROS_WARN_STREAM_THROTTLE(5.0, "TF lookup failed: " << ex.what());
    }

    rotation_matrix.col(0) = x_axis;
    rotation_matrix.col(1) = y_axis;
    rotation_matrix.col(2) = z_axis;
    if (rotation_matrix.determinant() < 0.0f) rotation_matrix.col(2) *= -1.0f;
    return true;
  }

  // Create and publish 3D detection message and marker
  void createAndPublishDetection(
      vision_msgs::Detection3DArray& detections3d_msg,
      visualization_msgs::MarkerArray& object_markers,
      visualization_msgs::MarkerArray& plane_markers,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const Eigen::Vector4f& centroid, const Eigen::Matrix3f& rotation_matrix,
      const std::vector<vision_msgs::ObjectHypothesisWithPose>& results,
      const std_msgs::Header& header, const double& duration,
      const std::string& frame_id) {
    if (cloud->points.empty()) {
      return;
    }
    Eigen::Matrix3f rot = rotation_matrix;  // Create copy
    Eigen::Quaternionf q(rot);
    // Create TransformStamped message and publish to object_transform_updates
    // topic
    geometry_msgs::TransformStamped transform_msg;
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
    object_transform_pub_.publish(transform_msg);
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
  ros::init(argc, argv, "process_tracker_with_cloud");
  ProcessTrackerWithCloud tracker;
  ros::spin();
  return 0;
}

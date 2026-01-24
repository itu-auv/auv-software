/**
 * @file object_plane_fitter_node.cpp
 * @brief ROS 1 node for estimating object pose via RANSAC plane fitting
 *
 * This node subscribes to depth images, camera info, and YOLO detections,
 * then uses RANSAC to fit planes to detected objects and publishes their poses.
 */

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vision_msgs/Detection2DArray.h>

#include <Eigen/Geometry>
#include <cmath>
#include <map>
#include <mutex>
#include <string>
#include <vector>

class ObjectPlaneFitter {
 public:
  ObjectPlaneFitter(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh),
        pnh_(pnh),
        camera_info_received_(false),
        tf_buffer_(ros::Duration(10.0)),
        tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    // Load parameters
    // Use -1 to indicate "all classes" (no filtering)
    pnh_.param<int>("target_class_id", target_class_id_, 4);
    pnh_.param<double>("ransac_distance_threshold", ransac_distance_threshold_,
                       0.03);
    pnh_.param<int>("ransac_max_iterations", ransac_max_iterations_, 1000);
    pnh_.param<double>("depth_scale_factor", depth_scale_factor_, 1.0);
    pnh_.param<double>("min_inlier_ratio", min_inlier_ratio_, 0.5);
    pnh_.param<bool>("publish_debug", publish_debug_, true);
    pnh_.param<std::string>("output_frame_id", output_frame_id_, "");
    pnh_.param<std::string>("camera_optical_frame", camera_optical_frame_,
                            "taluy/camera_depth_optical_frame");
    pnh_.param<std::string>("base_link_frame", base_link_frame_,
                            "taluy/base_link");

    // Publishers
    object_transform_pub_ = nh_.advertise<geometry_msgs::TransformStamped>(
        "object_transform_updates", 10);
    if (publish_debug_) {
      debug_cloud_pub_ =
          nh_.advertise<sensor_msgs::PointCloud2>("debug/cropped_cloud", 1);
      debug_inlier_cloud_pub_ =
          nh_.advertise<sensor_msgs::PointCloud2>("debug/inlier_cloud", 1);
    }

    // Camera info subscriber (separate, latched)
    camera_info_sub_ = nh_.subscribe(
        "camera_info", 1, &ObjectPlaneFitter::cameraInfoCallback, this);

    // Synchronized subscribers for depth and detections
    depth_sub_.subscribe(nh_, "depth", 1);
    detection_sub_.subscribe(nh_, "detections", 10);

    // ApproximateTime synchronizer
    sync_.reset(new Sync(SyncPolicy(10), depth_sub_, detection_sub_));
    sync_->registerCallback(
        boost::bind(&ObjectPlaneFitter::syncCallback, this, _1, _2));

    ROS_INFO("[ObjectPlaneFitter] Node initialized");
    ROS_INFO("[ObjectPlaneFitter] Target class ID: %s",
             target_class_id_ < 0 ? "(all)"
                                  : std::to_string(target_class_id_).c_str());
    ROS_INFO("[ObjectPlaneFitter] RANSAC threshold: %.4f m, max iterations: %d",
             ransac_distance_threshold_, ransac_max_iterations_);
  }

 private:
  void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_info_ = *msg;
    camera_info_received_ = true;
    ROS_INFO_ONCE("[ObjectPlaneFitter] Camera info received: %dx%d", msg->width,
                  msg->height);
  }

  void syncCallback(
      const sensor_msgs::Image::ConstPtr& depth_msg,
      const vision_msgs::Detection2DArray::ConstPtr& detections_msg) {
    // Check if camera info is available
    sensor_msgs::CameraInfo cam_info;
    {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      if (!camera_info_received_) {
        ROS_WARN_THROTTLE(5.0,
                          "[ObjectPlaneFitter] Waiting for camera info...");
        return;
      }
      cam_info = camera_info_;
    }

    // Convert depth image
    cv_bridge::CvImageConstPtr cv_depth;
    try {
      cv_depth = cv_bridge::toCvShare(depth_msg);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("[ObjectPlaneFitter] cv_bridge exception: %s", e.what());
      return;
    }

    // Get camera intrinsics
    double fx = cam_info.K[0];
    double fy = cam_info.K[4];
    double cx = cam_info.K[2];
    double cy = cam_info.K[5];

    if (fx == 0.0 || fy == 0.0) {
      ROS_ERROR_THROTTLE(
          5.0,
          "[ObjectPlaneFitter] Invalid camera intrinsics (fx=%.2f, fy=%.2f)",
          fx, fy);
      return;
    }

    // Process each detection
    for (const auto& detection : detections_msg->detections) {
      // Filter by class if specified (target_class_id_ < 0 means accept all)
      if (target_class_id_ >= 0) {
        bool class_match = false;
        for (const auto& result : detection.results) {
          if (result.id == target_class_id_) {
            class_match = true;
            break;
          }
        }
        if (!class_match) {
          continue;
        }
      }

      // Get bounding box
      int bbox_cx = static_cast<int>(detection.bbox.center.x);
      int bbox_cy = static_cast<int>(detection.bbox.center.y);
      int bbox_w = static_cast<int>(detection.bbox.size_x);
      int bbox_h = static_cast<int>(detection.bbox.size_y);

      int x_min = std::max(0, bbox_cx - bbox_w / 2);
      int y_min = std::max(0, bbox_cy - bbox_h / 2);
      int x_max = std::min(static_cast<int>(cv_depth->image.cols),
                           bbox_cx + bbox_w / 2);
      int y_max = std::min(static_cast<int>(cv_depth->image.rows),
                           bbox_cy + bbox_h / 2);

      if (x_max <= x_min || y_max <= y_min) {
        ROS_WARN_THROTTLE(1.0, "[ObjectPlaneFitter] Invalid bounding box");
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
        ROS_WARN_THROTTLE(
            1.0, "[ObjectPlaneFitter] Not enough valid points in ROI: %zu",
            cloud->points.size());
        continue;
      }

      cloud->width = cloud->points.size();
      cloud->height = 1;
      cloud->is_dense = false;

      // Publish debug cropped cloud
      if (publish_debug_ && debug_cloud_pub_.getNumSubscribers() > 0) {
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header = depth_msg->header;
        debug_cloud_pub_.publish(cloud_msg);
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
        ROS_WARN_THROTTLE(1.0,
                          "[ObjectPlaneFitter] RANSAC failed to find a plane");
        continue;
      }

      // Check inlier ratio
      double inlier_ratio =
          static_cast<double>(inliers->indices.size()) / cloud->points.size();
      if (inlier_ratio < min_inlier_ratio_) {
        ROS_WARN_THROTTLE(
            1.0, "[ObjectPlaneFitter] Inlier ratio too low: %.2f < %.2f",
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
        int detection_id = detection.results[0].id;
        auto it = id_to_prop_name_.find(detection_id);
        if (it != id_to_prop_name_.end()) {
          base_frame_id = it->second;
        }
      }
      std::string frame_id = base_frame_id + "_closest";

      // Publish object transform (same pattern as process_tracker_with_cloud)
      geometry_msgs::TransformStamped transform_msg;
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
      object_transform_pub_.publish(transform_msg);

      // Publish debug visualizations
      if (publish_debug_) {
        publishDebugVisualizations(depth_msg->header, inliers, cloud);
      }

      ROS_DEBUG(
          "[ObjectPlaneFitter] Plane fit: normal=(%.3f, %.3f, %.3f), "
          "centroid=(%.3f, %.3f, %.3f), inliers=%zu (%.1f%%)",
          normal.x(), normal.y(), normal.z(), centroid.x(), centroid.y(),
          centroid.z(), inliers->indices.size(), inlier_ratio * 100.0);
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
        ROS_ERROR_ONCE("[ObjectPlaneFitter] Unsupported depth image type: %d",
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
      geometry_msgs::TransformStamped tf_msg =
          tf_buffer_.lookupTransform(camera_optical_frame_, base_link_frame_,
                                     ros::Time(0), ros::Duration(0.05));
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
      ROS_WARN_STREAM_THROTTLE(5.0, "[ObjectPlaneFitter] TF lookup failed: "
                                        << ex.what() << ". Using fallback.");
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
      const std_msgs::Header& header, const pcl::PointIndices::Ptr& inliers,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    // Publish inlier point cloud
    if (debug_inlier_cloud_pub_.getNumSubscribers() > 0) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      pcl::ExtractIndices<pcl::PointXYZ> extract;
      extract.setInputCloud(cloud);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);

      sensor_msgs::PointCloud2 inlier_cloud_msg;
      pcl::toROSMsg(*inlier_cloud, inlier_cloud_msg);
      inlier_cloud_msg.header = header;
      debug_inlier_cloud_pub_.publish(inlier_cloud_msg);
    }
  }

  // ROS handles
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Publishers
  ros::Publisher object_transform_pub_;
  ros::Publisher debug_cloud_pub_;
  ros::Publisher debug_inlier_cloud_pub_;

  // Subscribers
  ros::Subscriber camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<vision_msgs::Detection2DArray> detection_sub_;

  // Synchronization
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, vision_msgs::Detection2DArray>
      SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  boost::shared_ptr<Sync> sync_;

  // Camera info
  sensor_msgs::CameraInfo camera_info_;
  bool camera_info_received_;
  std::mutex camera_info_mutex_;

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
  std::map<int, std::string> id_to_prop_name_ = {
      {0, "gate_sawfish_link"}, {1, "gate_shark_link"},
      {2, "red_pipe_link"},     {3, "white_pipe_link"},
      {4, "torpedo_map_link"},  {5, "torpedo_hole_link"},
      {6, "bin_whole_link"},    {7, "octagon_link"}};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "object_plane_fitter");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ObjectPlaneFitter node(nh, pnh);

  ros::spin();
  return 0;
}

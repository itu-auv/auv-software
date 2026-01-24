/**
 * @file object_plane_fitter_node.cpp
 * @brief ROS 1 node for estimating object pose via RANSAC plane fitting
 *
 * This node subscribes to depth images, camera info, and YOLO detections,
 * then uses RANSAC to fit planes to detected objects and publishes their poses.
 */

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <image_transport/image_transport.h>
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
#include <shape_msgs/Plane.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <vision_msgs/Detection2DArray.h>
#include <visualization_msgs/Marker.h>

#include <Eigen/Geometry>
#include <cmath>
#include <mutex>
#include <string>
#include <vector>

class ObjectPlaneFitter {
 public:
  ObjectPlaneFitter(ros::NodeHandle& nh, ros::NodeHandle& pnh)
      : nh_(nh), pnh_(pnh), camera_info_received_(false) {
    // Load parameters
    // Use -1 to indicate "all classes" (no filtering)
    pnh_.param<int>("target_class_id", target_class_id_, 4);
    pnh_.param<double>("ransac_distance_threshold", ransac_distance_threshold_,
                       0.01);
    pnh_.param<int>("ransac_max_iterations", ransac_max_iterations_, 1000);
    pnh_.param<double>("depth_scale_factor", depth_scale_factor_, 1.0);
    pnh_.param<double>("min_inlier_ratio", min_inlier_ratio_, 0.5);
    pnh_.param<bool>("publish_debug", publish_debug_, true);
    pnh_.param<std::string>("output_frame_id", output_frame_id_, "");

    // Publishers
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("object/pose", 1);
    plane_pub_ = nh_.advertise<shape_msgs::Plane>("object/plane_equation", 1);

    if (publish_debug_) {
      debug_cloud_pub_ =
          nh_.advertise<sensor_msgs::PointCloud2>("debug/cropped_cloud", 1);
      debug_marker_pub_ =
          nh_.advertise<visualization_msgs::Marker>("debug/plane_marker", 1);
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
      float d = coefficients->values[3];

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

      // Ensure normal points towards camera (negative Z in camera frame)
      if (normal.z() > 0) {
        normal = -normal;
        // Also flip d for consistency
        d = -d;
      }

      // Compute quaternion from normal
      // Align Z-axis with the plane normal
      Eigen::Quaternionf quat = computeQuaternionFromNormal(normal);

      // Publish pose
      geometry_msgs::PoseStamped pose_msg;
      pose_msg.header = depth_msg->header;
      if (!output_frame_id_.empty()) {
        pose_msg.header.frame_id = output_frame_id_;
      }
      pose_msg.pose.position.x = centroid.x();
      pose_msg.pose.position.y = centroid.y();
      pose_msg.pose.position.z = centroid.z();
      pose_msg.pose.orientation.x = quat.x();
      pose_msg.pose.orientation.y = quat.y();
      pose_msg.pose.orientation.z = quat.z();
      pose_msg.pose.orientation.w = quat.w();
      pose_pub_.publish(pose_msg);

      // Publish plane equation
      shape_msgs::Plane plane_msg;
      plane_msg.coef[0] = a;
      plane_msg.coef[1] = b;
      plane_msg.coef[2] = c;
      plane_msg.coef[3] = d;
      plane_pub_.publish(plane_msg);

      // Publish debug visualizations
      if (publish_debug_) {
        publishDebugVisualizations(depth_msg->header, inliers, cloud, centroid,
                                   normal);
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
    // We want to find a rotation that aligns the Z-axis with the normal
    Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);

    // Handle edge case where normal is already aligned with Z
    float dot = z_axis.dot(normal);
    if (std::abs(dot - 1.0f) < 1e-6f) {
      return Eigen::Quaternionf::Identity();
    }
    if (std::abs(dot + 1.0f) < 1e-6f) {
      // 180-degree rotation around X-axis
      return Eigen::Quaternionf(0.0f, 1.0f, 0.0f, 0.0f);
    }

    // Compute rotation axis and angle
    Eigen::Vector3f axis = z_axis.cross(normal);
    axis.normalize();

    float angle = std::acos(dot);

    Eigen::AngleAxisf rotation(angle, axis);
    return Eigen::Quaternionf(rotation);
  }

  void publishDebugVisualizations(
      const std_msgs::Header& header, const pcl::PointIndices::Ptr& inliers,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const Eigen::Vector3f& centroid, const Eigen::Vector3f& normal) {
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

    // Publish plane marker
    if (debug_marker_pub_.getNumSubscribers() > 0) {
      visualization_msgs::Marker marker;
      marker.header = header;
      marker.ns = "plane_fit";
      marker.id = 0;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;

      marker.pose.position.x = centroid.x();
      marker.pose.position.y = centroid.y();
      marker.pose.position.z = centroid.z();

      Eigen::Quaternionf quat = computeQuaternionFromNormal(normal);
      marker.pose.orientation.x = quat.x();
      marker.pose.orientation.y = quat.y();
      marker.pose.orientation.z = quat.z();
      marker.pose.orientation.w = quat.w();

      // Thin, wide plane
      marker.scale.x = 0.3;
      marker.scale.y = 0.3;
      marker.scale.z = 0.005;

      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.7f;

      marker.lifetime = ros::Duration(0.5);

      debug_marker_pub_.publish(marker);
    }
  }

  // ROS handles
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  // Publishers
  ros::Publisher pose_pub_;
  ros::Publisher plane_pub_;
  ros::Publisher debug_cloud_pub_;
  ros::Publisher debug_marker_pub_;
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

  // Parameters
  int target_class_id_;  // -1 means accept all classes
  double ransac_distance_threshold_;
  int ransac_max_iterations_;
  double depth_scale_factor_;
  double min_inlier_ratio_;
  bool publish_debug_;
  std::string output_frame_id_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "object_plane_fitter");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  ObjectPlaneFitter node(nh, pnh);

  ros::spin();
  return 0;
}

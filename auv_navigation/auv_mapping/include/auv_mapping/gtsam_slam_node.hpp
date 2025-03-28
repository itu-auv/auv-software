#pragma once

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>

// GTSAM headers
#include <auv_msgs/LandmarkDetection.h>  // Assuming this message type exists
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/isam2/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/symbolic/Symbol.h>

#include <map>
#include <string>

namespace auv_mapping {

class GtsamSlamNode {
 public:
  GtsamSlamNode(ros::NodeHandle& nh);
  ~GtsamSlamNode() = default;

 private:
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void landmarkCallback(const auv_msgs::LandmarkDetection::ConstPtr& msg);
  void optimize();
  void publishTransforms();

  ros::NodeHandle nh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber landmark_sub_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;

  gtsam::ISAM2 isam2_;
  gtsam::NonlinearFactorGraph graph_;
  gtsam::Values initial_estimates_;
  gtsam::Values current_estimates_;

  gtsam::Symbol robot_pose_;
  std::map<std::string, gtsam::Symbol>
      landmark_keys_;  // Map from landmark ID string to GTSAM symbol
  unsigned long landmark_key_index_ =
      0;  // Counter for assigning new landmark keys ('l0', 'l1', ...)

  nav_msgs::Odometry last_odom_pose_;
  struct PoseKeyInfo {
    gtsam::Symbol key;
    nav_msgs::Odometry pose;  // The odom pose when this key was added
  };
  PoseKeyInfo last_graph_pose_info_;
  bool is_initialized_ = false;

  // Noise models - Placeholders for now, will need to be configured
  gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr odometry_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr landmark_noise_;

  // Parameters
  std::string map_frame_id_;
  std::string odom_frame_id_;
  std::string base_frame_id_;
  double odom_threshold_dist_;
  double odom_threshold_angle_;
};

}  // namespace auv_mapping

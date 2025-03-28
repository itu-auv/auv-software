#include "auv_mapping/gtsam_slam_node.hpp"

#include <geometry_msgs/TransformStamped.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>  // Not strictly needed for iSAM2, but good to have
#include <gtsam/nonlinear/Marginals.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>  // For potential future use

namespace auv_mapping {

// Helper function to convert ROS Pose to GTSAM Pose3
gtsam::Pose3 rosPoseToGtsamPose3(const geometry_msgs::Pose& pose) {
  return gtsam::Pose3(
      gtsam::Rot3::Quaternion(pose.orientation.w, pose.orientation.x,
                              pose.orientation.y, pose.orientation.z),
      gtsam::Point3(pose.position.x, pose.position.y, pose.position.z));
}

// Helper function to convert GTSAM Pose3 to ROS Pose
geometry_msgs::Pose gtsamPose3ToRosPose(const gtsam::Pose3& pose) {
  geometry_msgs::Pose ros_pose;
  ros_pose.position.x = pose.x();
  ros_pose.position.y = pose.y();
  ros_pose.position.z = pose.z();
  ros_pose.orientation.w = pose.rotation().quaternion().w();
  ros_pose.orientation.x = pose.rotation().quaternion().x();
  ros_pose.orientation.y = pose.rotation().quaternion().y();
  ros_pose.orientation.z = pose.rotation().quaternion().z();
  return ros_pose;
}

// Constructor for GtsamSlamNode
GtsamSlamNode::GtsamSlamNode(ros::NodeHandle& nh)
    : nh_(nh),
      isam2_S()  // Default ISAM2 parameters
      ,
      robot_pose_('x')  // Using 'x' as base key for robot poses
{
  // --- Parameters ---
  // Store thresholds as member variables
  nh_.param<double>("odom_threshold_dist", odom_threshold_dist_,
                    0.5);  // meters
  nh_.param<double>("odom_threshold_angle", odom_threshold_angle_,
                    0.2);  // radians
  nh_.param<std::string>("map_frame_id", map_frame_id_, "map");
  nh_.param<std::string>("odom_frame_id", odom_frame_id_, "odom");
  nh_.param<std::string>("base_frame_id", base_frame_id_,
                         "base_link");  // Check TF tree

  std::string odom_topic, landmark_topic;
  nh_.param<std::string>("odom_topic", odom_topic,
                         "/taluy/odometry");  // Default to EKF output
  nh_.param<std::string>("landmark_topic", landmark_topic,
                         "/taluy/slam/landmark_detections");

  odom_sub_ = nh_.subscribe(odom_topic, 10, &GtsamSlamNode::odomCallback, this);
  landmark_sub_ =
      nh_.subscribe(landmark_topic, 10, &GtsamSlamNode::landmarkCallback,
                    this);  // Placeholder callback

  // Placeholder noise models - Tune these!
  // TODO: Read noise parameters from ROS params
  prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
  odometry_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1).finished());
  landmark_noise_ = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(3) << 0.1, 0.1, 0.1)
          .finished());  // Assuming 3D landmarks (x, y, z)

  robot_pose_ = gtsam::Symbol('x', 0);  // Initialize robot pose symbol index
}

void GtsamSlamNode::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  ROS_DEBUG_STREAM("Odom message received: " << msg->header.stamp.toSec());
  last_odom_pose_ = *msg;  // Store the latest odometry

  if (!is_initialized_) {
    ROS_INFO("Initializing GTSAM graph with first pose.");
    gtsam::Pose3 initial_pose = rosPoseToGtsamPose3(msg->pose.pose);

    // Add prior factor
    graph_.add(gtsam::PriorFactor<gtsam::Pose3>(robot_pose_, initial_pose,
                                                prior_noise_));

    // Add initial estimate
    initial_estimates_.insert(robot_pose_, initial_pose);

    // Store info about this first graph pose
    last_graph_pose_info_.key = robot_pose_;
    last_graph_pose_info_.pose = *msg;

    is_initialized_ = true;
    optimize();  // Optimize with the first pose and prior
  } else {
    // Calculate relative motion since last graph update
    gtsam::Pose3 prev_graph_odom_gtsam_pose =
        rosPoseToGtsamPose3(last_graph_pose_info_.pose.pose.pose);
    gtsam::Pose3 current_odom_gtsam_pose = rosPoseToGtsamPose3(msg->pose.pose);
    // Calculate relative motion in the odom frame since the last pose was added
    // to the graph
    gtsam::Pose3 relative_motion_odom =
        prev_graph_odom_gtsam_pose.between(current_odom_gtsam_pose);

    double dist_moved = relative_motion_odom.translation().norm();
    double angle_moved =
        relative_motion_odom.rotation().axisAngle().second;  // Angle in radians

    if (dist_moved > odom_threshold_dist_ ||
        angle_moved > odom_threshold_angle_) {
      ROS_INFO("Adding new pose node. Dist: %.2f, Angle: %.2f", dist_moved,
               angle_moved);

      // Increment the key for the new pose
      gtsam::Symbol prev_pose_key = last_graph_pose_info_.key;
      robot_pose_ = gtsam::Symbol(robot_pose_.chr(), robot_pose_.index() + 1);

      // Get the previous optimized pose from the current estimates
      gtsam::Pose3 prev_optimized_pose =
          current_estimates_.at<gtsam::Pose3>(prev_pose_key);

      // Calculate the initial estimate for the new pose by composing the
      // relative motion onto the *optimized* previous pose.
      gtsam::Pose3 new_pose_estimate =
          prev_optimized_pose * relative_motion_odom;
      initial_estimates_.insert(robot_pose_, new_pose_estimate);

      // Add the BetweenFactor (odometry factor) to the graph
      // Use the relative motion calculated from the raw odometry readings
      graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
          prev_pose_key, robot_pose_, relative_motion_odom, odometry_noise_));

      // Update the info for the latest pose added to the graph
      last_graph_pose_info_.key = robot_pose_;
      last_graph_pose_info_.pose = *msg;

      optimize();  // Trigger optimization
    } else {
      ROS_DEBUG("Movement below threshold. Dist: %.2f, Angle: %.2f", dist_moved,
                angle_moved);
      // Optional: If not adding a node, still publish the latest transform
      // based on current estimates This provides smoother updates between node
      // additions.
      publishTransforms();
    }
  }
}

// Updated signature and implementation
void GtsamSlamNode::landmarkCallback(
    const auv_msgs::LandmarkDetection::ConstPtr& msg) {
  ROS_DEBUG_STREAM("Landmark message received: " << msg->id << " at "
                                                 << msg->header.stamp.toSec());

  if (!is_initialized_) {
    ROS_WARN("Received landmark before odometry initialization, skipping.");
    return;
  }

  // --- Data Association ---
  // For simplicity, associate with the most recently added robot pose node.
  // A more robust approach would find the node closest in time to
  // msg->header.stamp.
  gtsam::Symbol robot_key = last_graph_pose_info_.key;

  // --- Get Landmark Info ---
  std::string landmark_id = msg->id;
  // Assuming msg->pose.pose is the relative pose of the landmark w.r.t. the
  // robot's base_link
  gtsam::Pose3 relative_landmark_pose = rosPoseToGtsamPose3(msg->pose.pose);
  // TODO: Use covariance from msg->pose.covariance if available for
  // landmark_noise_

  gtsam::Symbol landmark_key;
  bool is_new_landmark =
      landmark_keys_.find(landmark_id) == landmark_keys_.end();

  if (is_new_landmark) {
    ROS_INFO("New landmark detected: %s", landmark_id.c_str());
    // Generate a new symbol for the landmark
    landmark_key = gtsam::Symbol('l', landmark_key_index_++);
    landmark_keys_[landmark_id] = landmark_key;

    // Calculate initial estimate for the new landmark's pose in the map frame
    if (!current_estimates_.exists(robot_key)) {
      ROS_ERROR(
          "Robot pose key %s does not exist in current estimates. Cannot "
          "initialize landmark %s.",
          gtsam::DefaultKeyFormatter(robot_key).c_str(), landmark_id.c_str());
      return;
    }
    gtsam::Pose3 robot_pose_map =
        current_estimates_.at<gtsam::Pose3>(robot_key);
    gtsam::Pose3 landmark_pose_map_estimate =
        robot_pose_map * relative_landmark_pose;
    initial_estimates_.insert(landmark_key, landmark_pose_map_estimate);
    ROS_INFO("Added initial estimate for landmark %s at key %s",
             landmark_id.c_str(),
             gtsam::DefaultKeyFormatter(landmark_key).c_str());

  } else {
    landmark_key = landmark_keys_[landmark_id];
    ROS_DEBUG("Re-observing landmark %s with key %s", landmark_id.c_str(),
              gtsam::DefaultKeyFormatter(landmark_key).c_str());
  }

  // --- Add Factor ---
  // Add a BetweenFactor assuming the measurement is a full Pose3 relative to
  // the robot.
  // TODO: Verify this factor type matches the actual landmark sensor/message.
  // If it's bearing/range or pixel coordinates, use a different factor (e.g.,
  // BearingRangeFactor, GenericProjectionFactor).
  graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(
      robot_key, landmark_key, relative_landmark_pose, landmark_noise_));
  ROS_DEBUG("Added landmark factor between %s and %s",
            gtsam::DefaultKeyFormatter(robot_key).c_str(),
            gtsam::DefaultKeyFormatter(landmark_key).c_str());

  // --- Optimize ---
  // Optimization is triggered only when a new pose node is added in
  // odomCallback, but we could optionally trigger it here too if landmarks are
  // very informative or infrequent. optimize();
}

void GtsamSlamNode::optimize() {
  ROS_INFO_STREAM("Optimizing iSAM2...");
  try {
    isam2_.update(graph_, initial_estimates_);
    current_estimates_ = isam2_.calculateEstimate();
    graph_.resize(0);  // Clear the graph factors that were just added
    initial_estimates_
        .clear();  // Clear the initial estimates used for this update
    ROS_INFO_STREAM("Optimization finished.");
    publishTransforms();
  } catch (const gtsam::IndeterminantLinearSystemException& e) {
    ROS_ERROR_STREAM("GTSAM IndeterminantLinearSystemException: "
                     << e.what() << ". Graph size: " << graph_.size()
                     << ", Estimates size: " << initial_estimates_.size());
    // Handle error, maybe reset or skip update
    graph_.resize(0);  // Clear potentially problematic factors
    initial_estimates_.clear();
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("GTSAM exception during optimization: " << e.what());
    graph_.resize(0);
    initial_estimates_.clear();
  }
}

void GtsamSlamNode::publishTransforms() {
  if (!is_initialized_ ||
      !current_estimates_.exists(last_graph_pose_info_.key)) {
    ROS_WARN_THROTTLE(
        5.0, "Cannot publish transform, graph not initialized or key missing.");
    return;  // Not ready to publish yet
  }

  // Get the latest optimized pose from GTSAM estimates (pose of the robot in
  // the map frame at the time the last node was added)
  gtsam::Pose3 latest_optimized_gtsam_pose =
      current_estimates_.at<gtsam::Pose3>(last_graph_pose_info_.key);

  // Get the corresponding raw odometry pose (pose of the robot in the odom
  // frame at the time the last node was added)
  gtsam::Pose3 last_graph_odom_gtsam_pose =
      rosPoseToGtsamPose3(last_graph_pose_info_.pose.pose.pose);

  // Calculate the correction: T_map_odom = T_map_base * T_odom_base^-1
  // T_map_base is latest_optimized_gtsam_pose
  // T_odom_base is last_graph_odom_gtsam_pose
  gtsam::Pose3 map_to_odom_transform =
      latest_optimized_gtsam_pose * last_graph_odom_gtsam_pose.inverse();

  // Convert to ROS TransformStamped
  geometry_msgs::TransformStamped transform_stamped;
  // Use the timestamp of the odom message that triggered the last graph update
  transform_stamped.header.stamp = last_graph_pose_info_.pose.header.stamp;
  transform_stamped.header.frame_id = map_frame_id_;
  transform_stamped.child_frame_id = odom_frame_id_;
  transform_stamped.transform.translation.x = map_to_odom_transform.x();
  transform_stamped.transform.translation.y = map_to_odom_transform.y();
  transform_stamped.transform.translation.z = map_to_odom_transform.z();
  transform_stamped.transform.rotation.w =
      map_to_odom_transform.rotation().quaternion().w();
  transform_stamped.transform.rotation.x =
      map_to_odom_transform.rotation().quaternion().x();
  transform_stamped.transform.rotation.y =
      map_to_odom_transform.rotation().quaternion().y();
  transform_stamped.transform.rotation.z =
      map_to_odom_transform.rotation().quaternion().z();

  tf_broadcaster_.sendTransform(transform_stamped);
  ROS_DEBUG_STREAM("Published map->odom transform at "
                   << transform_stamped.header.stamp.toSec());
}

}  // namespace auv_mapping

int main(int argc, char** argv) {
  ros::init(argc, argv, "gtsam_slam_node");
  ros::NodeHandle nh("~");  // Use private node handle for parameters
  auv_mapping::GtsamSlamNode slam_node(nh);
  ROS_INFO_STREAM("gtsam_slam_node started.");
  ros::spin();
  return 0;
}

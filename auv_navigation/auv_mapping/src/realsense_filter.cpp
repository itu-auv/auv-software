#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h> // Changed Float64 and Float32MultiArray to Float32
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath> // For std::sqrt, std::acos, M_PI, std::isfinite
#include <limits> // For std::numeric_limits

using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

class PointCloudFilterNode {
public:
  PointCloudFilterNode(ros::NodeHandle& nh) {
    // Hardcoded topic names
    pc_topic_ = "/camera/depth/color/points";
    alt_topic_ = "/taluy/sensors/dvl/altitude";
    depth_topic_ = "/taluy/sensors/external_pressure_sensor/depth";

    ROS_INFO_ONCE("PointCloudFilterNode started."); // Node start message

    // setup subscribers and publisher
    pc_sub_ = nh.subscribe(pc_topic_, 1, &PointCloudFilterNode::cloudCallback, this);
    alt_sub_ = nh.subscribe(alt_topic_, 1, &PointCloudFilterNode::altCallback, this);
    depth_sub_ = nh.subscribe(depth_topic_, 1, &PointCloudFilterNode::depthCallback, this);
    pc_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/points_filtered", 1);

    alt_received_ = false;
    depth_received_ = false;
  }

private:
  // topics
  std::string pc_topic_, alt_topic_, depth_topic_;

  // ros interfaces
  ros::Subscriber pc_sub_, alt_sub_, depth_sub_;
  ros::Publisher pc_pub_;

  // latest sensor data
  float altitude_; // Changed type to float
  float depth_value_; // Changed from depth_array_ to single float
  bool alt_received_, depth_received_;

  void altCallback(const std_msgs::Float32::ConstPtr& msg) { // Changed message type
    altitude_ = msg->data;
    alt_received_ = true;
    ROS_INFO("Altitude received: %f", altitude_); // Altitude message received
  }

  void depthCallback(const std_msgs::Float32::ConstPtr& msg) { // Changed message type
    depth_value_ = msg->data; // Assign to single value
    depth_received_ = true;
    ROS_INFO("Depth received: %f", depth_value_); // Changed log message and format
  }

  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud_msg) {
    ROS_INFO("PointCloud received."); // PointCloud message received
    if (!alt_received_ || !depth_received_) {
      ROS_WARN_THROTTLE(5.0, "Waiting for altitude and depth data...");
      return;
    }

    // convert to PCL
    PointCloud::Ptr pcl_cloud(new PointCloud);
    pcl::fromROSMsg(*cloud_msg, *pcl_cloud);

    // Apply Refraction Correction
    PointCloud::Ptr refraction_corrected_cloud(new PointCloud);
    refraction_corrected_cloud->header = pcl_cloud->header; // Copy header from original PCL cloud

    for (const auto& pt_in : pcl_cloud->points) {
        double x = pt_in.x;
        double y = pt_in.y;
        double z_in = pt_in.z;
        double corrected_z = z_in; // Default to original z if correction is not possible/applicable

        // Using epsilon for floating point comparisons to avoid division by zero for r
        double r = std::sqrt(x * x + z_in * z_in);

        if (r > std::numeric_limits<double>::epsilon()) {
            double val_for_acos = z_in / r;
            // Clamp val_for_acos to [-1.0, 1.0] to prevent acos domain errors due to precision issues
            if (val_for_acos < -1.0) val_for_acos = -1.0;
            if (val_for_acos > 1.0) val_for_acos = 1.0;

            double theta = std::acos(val_for_acos);
            double angle_deg = theta * 180.0 / M_PI; // M_PI should be defined in <cmath>
            double error_percent = -25.0 - 0.02 * angle_deg * angle_deg;
            
            double denominator = 1.0 + (error_percent / 100.0);
            // Avoid division by zero for denominator
            if (std::abs(denominator) > std::numeric_limits<double>::epsilon()) {
                double correction_factor = 1.0 / denominator;
                corrected_z = z_in * correction_factor;
            }
            // else: keep original z_in if denominator is effectively zero
        }
        // else: keep original z_in if r is effectively zero

        pcl::PointXYZ corrected_pt;
        corrected_pt.x = static_cast<float>(x);
        corrected_pt.y = static_cast<float>(y);
        corrected_pt.z = static_cast<float>(corrected_z);

        // Add point only if all coordinates are finite
        if (std::isfinite(corrected_pt.x) && std::isfinite(corrected_pt.y) && std::isfinite(corrected_pt.z)) {
            refraction_corrected_cloud->points.push_back(corrected_pt);
        }
    }

    refraction_corrected_cloud->width = refraction_corrected_cloud->points.size();
    refraction_corrected_cloud->height = 1;
    // is_dense is true if all points are finite. Since we check std::isfinite before adding,
    // all points in refraction_corrected_cloud are finite.
    refraction_corrected_cloud->is_dense = true; 


    PointCloud::Ptr temp_cloud(new PointCloud);

    // Manual filtering: keep points that satisfy all conditions
    // Filter from the refraction_corrected_cloud
    temp_cloud->header = refraction_corrected_cloud->header; // Use header from the corrected cloud
    
    for (const auto& pt : refraction_corrected_cloud->points) { // Iterate over corrected points
      // Filter based on Y coordinate using altitude and depth
      // Keep points below altitude (+offset) and above depth (-offset)
      // Also check if AUV depth itself is less than 5.0m
      if (pt.y <= altitude_ - 0.45 && pt.y >= depth_value_ + 0.30 && pt.z < 5.0) {
        temp_cloud->points.push_back(pt);
      }
    }
    temp_cloud->width = temp_cloud->points.size();
    temp_cloud->height = 1;
    temp_cloud->is_dense = false; // Keep original behavior for the final published cloud

    // publish
    sensor_msgs::PointCloud2 out_msg;
    pcl::toROSMsg(*temp_cloud, out_msg);
    out_msg.header = cloud_msg->header;
    pc_pub_.publish(out_msg);
  }

  // Removed getMinDepth and getDepthValue functions as they are no longer needed
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "pointcloud_filter_node");
  ros::NodeHandle nh;

  PointCloudFilterNode node(nh);
  ros::spin();

  return 0;
}

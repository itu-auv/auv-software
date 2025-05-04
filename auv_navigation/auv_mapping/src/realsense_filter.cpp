#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Float32.h> // Changed Float64 and Float32MultiArray to Float32
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

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

    PointCloud::Ptr temp_cloud(new PointCloud);

    // Manual filtering: keep points that satisfy all conditions
    temp_cloud->header = pcl_cloud->header;
    for (size_t i = 0; i < pcl_cloud->points.size(); ++i) {
      const auto& pt = pcl_cloud->points[i];
      // Filter based on Y coordinate using altitude and depth
      // Keep points below altitude (+offset) and above depth (-offset)
      // Also check if AUV depth itself is less than 5.0m
      if (pt.y <= altitude_ - 0.45 && pt.y >= depth_value_ + 0.30 && pt.z < 5.0) {
        temp_cloud->points.push_back(pt);
      }
    }
    temp_cloud->width = temp_cloud->points.size();
    temp_cloud->height = 1;
    temp_cloud->is_dense = false;

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

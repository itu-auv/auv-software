#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Range.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Header.h>
#include <std_srvs/SetBool.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <vector>

class SonarToPointCloud {
 public:
  SonarToPointCloud()
      : nh_("~"),
        tf_buffer_(),
        tf_listener_(tf_buffer_),
        services_active_(false),
        reference_z_(std::numeric_limits<double>::quiet_NaN()),
        point_count_since_last_detection_(0) {
    // Load parameters (private namespace)
    nh_.param("max_point_cloud_size", max_point_cloud_size_, 200000);
    nh_.param("min_valid_range", min_valid_range_, 0.3);
    nh_.param("max_valid_range", max_valid_range_, 100.0);
    nh_.param<std::string>("frame_id", frame_id_, "odom");
    nh_.param<std::string>("sonar_front_frame", sonar_front_frame_,
                           "taluy/base_link/sonar_front_link");
    nh_.param<std::string>("sonar_right_frame", sonar_right_frame_,
                           "taluy/base_link/sonar_right_link");
    nh_.param<std::string>("sonar_left_frame", sonar_left_frame_,
                           "taluy/base_link/sonar_left_link");

    // Publisher for the accumulated point cloud
    pc_publisher_ =
        nh_.advertise<sensor_msgs::PointCloud2>("/sonar/point_cloud", 10);

    // Services to start/stop mapping and to clear points
    start_service_ = nh_.advertiseService(
        "/sonar/start_mapping", &SonarToPointCloud::handleStartService, this);
    clear_service_ = nh_.advertiseService(
        "/sonar/clear_points", &SonarToPointCloud::handleClearService, this);

    ROS_INFO(
        "[SonarToPointCloud] Node initialized. Use /sonar/start_mapping and "
        "/sonar/clear_points services.");
  }

  void run() { ros::spin(); }

 private:
  // ---- Member variables ----
  ros::NodeHandle nh_;

  // TF2
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Publishers and subscribers
  ros::Publisher pc_publisher_;
  ros::Subscriber sonar_front_sub_;
  ros::Subscriber sonar_right_sub_;
  ros::Subscriber sonar_left_sub_;

  // Service servers
  ros::ServiceServer start_service_;
  ros::ServiceServer clear_service_;

  // Parameters
  int max_point_cloud_size_;
  double min_valid_range_;
  double max_valid_range_;
  std::string frame_id_;
  std::string sonar_front_frame_;
  std::string sonar_right_frame_;
  std::string sonar_left_frame_;

  // Accumulated data
  std::vector<geometry_msgs::Point> point_cloud_data_;
  double reference_z_;
  int point_count_since_last_detection_;
  bool services_active_;

  // Thread safety
  std::mutex mutex_;

  // ---- Service callbacks ----

  bool handleStartService(std_srvs::SetBool::Request& req,
                          std_srvs::SetBool::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (req.data)  // request to start mapping
    {
      if (!services_active_) {
        services_active_ = true;

        // Create subscribers
        sonar_front_sub_ = nh_.subscribe<sensor_msgs::Range>(
            "/taluy/sensors/sonar_front/data", 10,
            boost::bind(&SonarToPointCloud::sonarFrontCallback, this, _1));

        sonar_right_sub_ = nh_.subscribe<sensor_msgs::Range>(
            "/taluy/sensors/sonar_right/data", 10,
            boost::bind(&SonarToPointCloud::sonarRightCallback, this, _1));

        sonar_left_sub_ = nh_.subscribe<sensor_msgs::Range>(
            "/taluy/sensors/sonar_left/data", 10,
            boost::bind(&SonarToPointCloud::sonarLeftCallback, this, _1));

        ROS_INFO(
            "[SonarToPointCloud] Pool boundary detection started with all "
            "three sonars.");
        res.success = true;
        res.message = "Service started";
      } else {
        res.success = false;
        res.message = "Service is already running";
      }
    } else  // request to stop mapping
    {
      if (services_active_) {
        services_active_ = false;

        // Unregister subscribers
        sonar_front_sub_.shutdown();
        sonar_right_sub_.shutdown();
        sonar_left_sub_.shutdown();

        ROS_INFO("[SonarToPointCloud] Pool boundary detection stopped.");
        res.success = true;
        res.message = "Service stopped";
      } else {
        res.success = false;
        res.message = "Service is already stopped";
      }
    }
    return true;
  }

  bool handleClearService(std_srvs::SetBool::Request& req,
                          std_srvs::SetBool::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (req.data) {
      point_cloud_data_.clear();
      reference_z_ = std::numeric_limits<double>::quiet_NaN();
      point_count_since_last_detection_ = 0;
      ROS_INFO("[SonarToPointCloud] Point cloud data cleared.");

      publishPointCloud();

      res.success = true;
      res.message = "Point cloud cleared";
    } else {
      res.success = false;
      res.message = "No action taken";
    }
    return true;
  }

  // ---- Sonar callbacks ----

  void sonarFrontCallback(const sensor_msgs::Range::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    processSonarReading(msg, sonar_front_frame_);
  }

  void sonarRightCallback(const sensor_msgs::Range::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    processSonarReading(msg, sonar_right_frame_);
  }

  void sonarLeftCallback(const sensor_msgs::Range::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    processSonarReading(msg, sonar_left_frame_);
  }

  // ---- Processing logic ----

  void processSonarReading(const sensor_msgs::Range::ConstPtr& msg,
                           const std::string& sonar_frame) {
    if (!services_active_) return;

    if (!(msg->range > min_valid_range_ && msg->range < max_valid_range_))
      return;

    // Create a PointStamped in the sonar frame at (range, 0, 0)
    geometry_msgs::PointStamped sensor_point;
    sensor_point.header.stamp = msg->header.stamp;
    sensor_point.header.frame_id = sonar_frame;
    sensor_point.point.x = msg->range;
    sensor_point.point.y = 0.0;
    sensor_point.point.z = 0.0;

    // Transform into the odom frame
    geometry_msgs::PointStamped odom_point;
    if (!transformPointToOdom(sensor_point, odom_point)) return;

    // On first valid point, set reference Z
    if (std::isnan(reference_z_)) {
      reference_z_ = odom_point.point.z;
      ROS_INFO("[SonarToPointCloud] Reference Z set to: %f", reference_z_);
    }

    // Maintain buffer size
    if ((int)point_cloud_data_.size() >= max_point_cloud_size_) {
      point_cloud_data_.erase(point_cloud_data_.begin());
    }

    // Append the new point
    point_cloud_data_.push_back(odom_point.point);
    ++point_count_since_last_detection_;

    // Publish updated point cloud
    publishPointCloud();
  }

  bool transformPointToOdom(const geometry_msgs::PointStamped& in_point,
                            geometry_msgs::PointStamped& out_point) {
    try {
      geometry_msgs::TransformStamped transform_stamped =
          tf_buffer_.lookupTransform(frame_id_, in_point.header.frame_id,
                                     in_point.header.stamp, ros::Duration(0.1));

      tf2::doTransform(in_point, out_point, transform_stamped);
      return true;
    } catch (tf2::TransformException& ex) {
      ROS_WARN_THROTTLE(10.0, "[SonarToPointCloud] TF lookup failed: %s",
                        ex.what());
      return false;
    }
  }

  void publishPointCloud() {
    sensor_msgs::PointCloud2 cloud_msg;
    cloud_msg.header.stamp = ros::Time::now();
    cloud_msg.header.frame_id = frame_id_;
    cloud_msg.height = 1;
    cloud_msg.width = static_cast<uint32_t>(point_cloud_data_.size());
    cloud_msg.is_dense = false;
    cloud_msg.is_bigendian = false;

    // Define XYZ fields
    sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
    modifier.setPointCloud2FieldsByString(1, "xyz");
    modifier.resize(point_cloud_data_.size());

    // Fill the data
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");

    for (const auto& pt : point_cloud_data_) {
      *iter_x = pt.x;
      *iter_y = pt.y;
      *iter_z = pt.z;
      ++iter_x;
      ++iter_y;
      ++iter_z;
    }

    pc_publisher_.publish(cloud_msg);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "sonar_to_pointcloud");
  SonarToPointCloud node;
  node.run();
  return 0;
}

#pragma once

#include <auv_msgs/TDOA.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

class TDOALocalizer {
  static constexpr auto c = 1500.0f;  // Speed of sound (m/s)

 public:
  TDOALocalizer(ros::NodeHandle& nh)
      : tf_buffer_(),
        tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    if (!loadHydrophonePos()) {
      ROS_ERROR("Failed to load hydrophone positions from TF.");
      ros::shutdown();
    }

    sub_ = nh.subscribe("acoustics/hydrophones/tdoa", 1,
                        &TDOALocalizer::callback, this);
    pub_ = nh.advertise<geometry_msgs::TransformStamped>("mapping/pinger", 1);

    transform_.header.frame_id = "odom";
    transform_.child_frame_id = "pinger";
    transform_.transform.rotation.x = 0.0;
    transform_.transform.rotation.y = 0.0;
    transform_.transform.rotation.z = 0.0;
    transform_.transform.rotation.w = 1.0;
  }

 private:
  ros::Subscriber sub_;
  ros::Publisher pub_;
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  geometry_msgs::TransformStamped transform_;

  std::array<std::array<double, 3>, 4> sensors_;

  // Read hydrophone positions from TF
  bool loadHydrophonePos() {
    const std::string parent = "taluy/base_link";
    const std::vector<std::string> names = {"hydrophone_1", "hydrophone_2",
                                            "hydrophone_3", "hydrophone_4"};

    for (size_t i = 0; i < names.size(); ++i) {
      geometry_msgs::TransformStamped tfst;
      try {
        tfst = tf_buffer_.lookupTransform(parent, names[i], ros::Time(0),
                                          ros::Duration(1.0)  // 1 saniye bekle
        );
      } catch (tf2::TransformException& ex) {
        ROS_ERROR("TF lookup for %s failed: %s", names[i].c_str(), ex.what());
        return false;
      }

      sensors_[i][0] = tfst.transform.translation.x;
      sensors_[i][1] = tfst.transform.translation.y;
      sensors_[i][2] = tfst.transform.translation.z;
      ROS_INFO("Hydrophone %zu at [%.2f, %.2f, %.2f]", i, sensors_[i][0],
               sensors_[i][1], sensors_[i][2]);
    }
    return true;
  }

  double distance(const std::array<double, 3>& a,
                  const std::array<double, 3>& b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  void callback(const auv_msgs::TDOA::ConstPtr& msg) {
    std::array<double, 4> tdoas_measured = {
        0.0, static_cast<double>(msg->td21) / 1e6,
        static_cast<double>(msg->td31) / 1e6,
        static_cast<double>(msg->td41) / 1e6};

    constexpr auto x_steps = 100U;
    constexpr auto y_steps = 100U;
    constexpr auto z_steps = 10U;
    constexpr auto x_min = -25.0, x_max = 25.0;
    constexpr auto y_min = -25.0, y_max = 25.0;
    constexpr auto z_min = -3.0, z_max = 0.0;

    const auto min_error = std::numeric_limits<double>::max();
    auto best_pos = std::array<double, 3>{0.0, 0.0, 0.0};

    for (int xi = 0; xi < x_steps; ++xi) {
      double x = x_min + xi * (x_max - x_min) / (x_steps - 1);
      for (int yi = 0; yi < y_steps; ++yi) {
        double y = y_min + yi * (y_max - y_min) / (y_steps - 1);
        for (int zi = 0; zi < z_steps; ++zi) {
          double z = z_min + zi * (z_max - z_min) / (z_steps - 1);
          auto test_point = std::array<double, 3>{x, y, z};

          std::array<double, 4> arrival_times_est;
          for (int i = 0; i < 4; ++i) {
            arrival_times_est[i] = distance(sensors_[i], test_point) / c;
          }

          std::array<double, 4> tdoas_est;
          for (int i = 0; i < 4; ++i) {
            tdoas_est[i] = arrival_times_est[i] - arrival_times_est[0];
          }

          auto error = 0.0;
          for (int i = 0; i < 4; ++i) {
            auto diff = tdoas_est[i] - tdoas_measured[i];
            error += diff * diff;
          }

          if (error < min_error) {
            min_error = error;
            best_pos = test_point;
          }
        }
      }
    }

    // 8) Transform Odom → Base_link
    geometry_msgs::TransformStamped tf_odom_base;
    try {
      tf_odom_base = tf_buffer_.lookupTransform(
          "odom", "taluy/base_link", ros::Time(0), ros::Duration(1.0));
    } catch (tf2::TransformException& ex) {
      ROS_ERROR("TF lookup (odom→base_link) failed: %s", ex.what());
      return;
    }

    geometry_msgs::TransformStamped tf_base_pinger;
    tf_base_pinger.header.stamp = tf_odom_base.header.stamp;
    tf_base_pinger.header.frame_id = "taluy/base_link";
    tf_base_pinger.child_frame_id = "pinger";
    tf_base_pinger.transform.translation.x = best_pos[0];
    tf_base_pinger.transform.translation.y = best_pos[1];
    tf_base_pinger.transform.translation.z = best_pos[2];
    tf_base_pinger.transform.rotation.x = 0.0;
    tf_base_pinger.transform.rotation.y = 0.0;
    tf_base_pinger.transform.rotation.z = 0.0;
    tf_base_pinger.transform.rotation.w = 1.0;

    geometry_msgs::TransformStamped tf_odom_pinger;
    tf2::doTransform(tf_base_pinger, tf_odom_pinger, tf_odom_base);

    pub_.publish(tf_odom_pinger);
  }
};

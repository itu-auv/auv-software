#pragma once

#include <auv_msgs/TDOA.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

class TDOALocalizer {
  static constexpr auto c = 1500.0f;  // Speed of sound (m/s)
 public:
  TDOALocalizer(ros::NodeHandle& nh) {
    if (!load_hydrophone_pos(nh)) {
      ROS_ERROR("Failed to load hydrophone positions.");
      ros::shutdown();
    }

    sub_ = nh.subscribe("acoustics/hydrophones/tdoa", 1,
                        &TDOALocalizer::callback, this);
    pub_ = nh.advertise<geometry_msgs::TransformStamped>("mapping/pinger", 1);

    transform.header.frame_id = "map";
    transform.child_frame_id = "pinger";
    transform.transform.rotation.x = 0.0;
    transform.transform.rotation.y = 0.0;
    transform.transform.rotation.z = 0.0;
    transform.transform.rotation.w = 1.0;
  }

 private:
  ros::Subscriber sub_;
  ros::Publisher pub_;
  geometry_msgs::TransformStamped transform;
  std::array<std::array<double, 3>, 4> sensors_;

  bool load_hydrophone_pos(ros::NodeHandle& nh) {
    std::vector<std::vector<double>> positions;
    if (!nh.getParam("~hydrophones", positions) || positions.size() != 4) {
      ROS_ERROR("Invalid or missing 'hydrophones' param.");
      return false;
    }

    for (std::size_t i = 0; i < 4; ++i) {
      if (positions[i].size() != 3) {
        ROS_ERROR("Hydrophone %zu does not have 3 coordinates.", i);
        return false;
      }
      sensors_[i] = {positions[i][0], positions[i][1], positions[i][2]};
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

    transform.header.stamp = ros::Time::now();
    transform.transform.translation.x = best_pos[0];
    transform.transform.translation.y = best_pos[1];
    transform.transform.translation.z = best_pos[2];

    pub_.publish(transform);
  }
};

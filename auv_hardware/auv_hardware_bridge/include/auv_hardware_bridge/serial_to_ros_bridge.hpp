#pragma once

#include <deque>
#include <mutex>
#include <string>

#include "ros/ros.h"
#include "std_msgs/UInt8MultiArray.h"

namespace auv_hardware {
class SerialToROSBridge {
 public:
  SerialToROSBridge(ros::NodeHandle &nh);

  void spin();

 private:
  void input_data_callback(const std_msgs::UInt8MultiArray::ConstPtr &msg);

  void handle_incoming_messages();

  void handle_outgoing_messages();

  int master_fd_;
  ros::Subscriber subscriber_;
  ros::Publisher publisher_;
  std::deque<std::string> message_queue_;
  std::mutex mutex_;
  char buffer_[1024];
};

};  // namespace auv_hardware

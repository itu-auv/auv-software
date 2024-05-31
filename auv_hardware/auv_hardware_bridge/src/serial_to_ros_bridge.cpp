#include "auv_hardware_bridge/serial_to_ros_bridge.hpp"
#include <boost/asio.hpp>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <pty.h>
#include <unistd.h>

namespace auv_hardware {

SerialToROSBridge::SerialToROSBridge(ros::NodeHandle &nh) {
  ros::NodeHandle nh_priv("~");
  const auto virtual_port =
      nh_priv.param<std::string>("virtual_port", "/tmp/vcom0");

  int slave_fd;
  if (openpty(&master_fd_, &slave_fd, nullptr, nullptr, nullptr) == -1) {
    ROS_ERROR_STREAM("Error creating pseudoterminal: " << strerror(errno));
    return;
  }

  // Create a symbolic link to the virtual serial port
  const auto slave_name = std::string{ttyname(slave_fd)};

  const auto result = symlink(slave_name.c_str(), virtual_port.c_str());

  if (result == -1) {
    ROS_ERROR_STREAM("Error creating symlink from: " << slave_name << " to: "
                                                     << virtual_port << ": "
                                                     << strerror(errno));
    return;
  }

  subscriber_ = nh.subscribe("incoming", 10,
                             &SerialToROSBridge::input_data_callback, this);
  publisher_ = nh.advertise<std_msgs::UInt8MultiArray>("outgoing", 10);

  ROS_INFO_STREAM("Serial to ROS bridge started. Port: "
                  << virtual_port << " topics: " << subscriber_.getTopic()
                  << " " << publisher_.getTopic());
}

void SerialToROSBridge::spin() {
  while (ros::ok()) {
    handle_incoming_messages();

    handle_outgoing_messages();

    ros::spinOnce();
  }
}

void SerialToROSBridge::input_data_callback(
    const std_msgs::UInt8MultiArray::ConstPtr &msg) {
  auto lock = std::scoped_lock<std::mutex>{mutex_};

  const auto message = std::string{msg->data.begin(), msg->data.end()};
  message_queue_.push_back(message);
}

void SerialToROSBridge::handle_incoming_messages() {
  auto lock = std::scoped_lock<std::mutex>{mutex_};

  while (!message_queue_.empty()) {
    const auto message = message_queue_.front();
    message_queue_.pop_front();

    const auto result = write(master_fd_, message.c_str(), message.length());
    if (result == -1) {
      ROS_ERROR_STREAM("Error writing to pseudoterminal: " << strerror(errno));
    }
  }
}

void SerialToROSBridge::handle_outgoing_messages() {
  fd_set read_fds;
  FD_ZERO(&read_fds);
  FD_SET(master_fd_, &read_fds);

  timeval timeout = {0, 100000}; // 0.1 seconds
  const int result =
      select(master_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);

  if (result <= 0 || !FD_ISSET(master_fd_, &read_fds)) {
    return;
  }

  const auto bytes_read = read(master_fd_, buffer_, sizeof(buffer_));

  if (bytes_read <= 0) {
    return;
  }

  auto msg = std_msgs::UInt8MultiArray{};
  msg.data.assign(buffer_, buffer_ + bytes_read);
  publisher_.publish(msg);
}

} // namespace auv_hardware

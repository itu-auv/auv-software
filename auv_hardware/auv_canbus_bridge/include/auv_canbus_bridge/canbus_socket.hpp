#pragma once

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <ros/ros.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace auv_hardware {

class CanbusSocket {
 public:
  CanbusSocket() : socket_fd_(-1) {}

  void initialize(const std::string &intarface_name) {
    socket_fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
      throw std::runtime_error("Failed to open CAN socket");
    }

    struct ifreq ifr;
    std::strncpy(ifr.ifr_name, intarface_name.c_str(), IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = '\0';

    const auto ioctl_status = ::ioctl(socket_fd_, SIOCGIFINDEX, &ifr);

    if (ioctl_status < 0) {
      ROS_ERROR_STREAM("Failed to get CAN interface index: " << intarface_name);
      close();
      throw std::runtime_error("Failed to get CAN interface index");
    }

    struct sockaddr_can addr = {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    const auto bind_status = ::bind(
        socket_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));

    if (bind_status < 0) {
      close();
      ROS_ERROR_STREAM(
          "Failed to bind CAN socket on interface: " << intarface_name);
      throw std::runtime_error("Failed to bind CAN socket");
    }
  }

  ~CanbusSocket() { close(); }

  template <size_t N>
  bool dispatch_message(const auv_hardware::canbus::ExtendedIdType id,
                        const std::array<uint8_t, N> &data) {
    struct can_frame frame = {};
    frame.can_id = id | CAN_EFF_FLAG;
    frame.can_dlc = N;

    std::memcpy(frame.data, data.data(), N);

    if (!is_connected()) {
      ROS_ERROR_STREAM("Failed to write CAN frame: (not connected)");
      return false;
    }

    {
      std::scoped_lock lock{socket_mutex_};
      const auto bytes_written = ::write(socket_fd_, &frame, sizeof(frame));

      if (bytes_written != sizeof(frame)) {
        ROS_ERROR_STREAM("Failed to write CAN frame");
        return false;
      }
    }

    return true;
  }

  bool dispatch_message(const auv_hardware::canbus::ExtendedIdType id,
                        const std::vector<uint8_t> &data) {
    struct can_frame frame = {};
    frame.can_id = id | CAN_EFF_FLAG;
    frame.can_dlc = data.size();

    std::memcpy(frame.data, data.data(), data.size());

    if (!is_connected()) {
      ROS_ERROR_STREAM("Failed to write CAN frame: (not connected)");
      return false;
    }

    {
      std::scoped_lock lock{socket_mutex_};
      const auto bytes_written = ::write(socket_fd_, &frame, sizeof(frame));

      if (bytes_written != sizeof(frame)) {
        ROS_ERROR_STREAM("Failed to write CAN frame");
        return false;
      }
    }

    return true;
  }

 protected:
  bool is_connected() const { return socket_fd_ >= 0; }

  void close() {
    if (!is_connected()) {
      return;
    }

    ::close(socket_fd_);
    socket_fd_ = -1;
  }

  int socket_fd_;
  std::mutex socket_mutex_;
};

}  // namespace auv_hardware

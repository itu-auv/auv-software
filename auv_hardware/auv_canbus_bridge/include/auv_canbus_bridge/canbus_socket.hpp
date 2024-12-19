#pragma once

#include <fcntl.h>
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
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace auv_hardware {
struct CanMessage {
  auv_hardware::canbus::ExtendedIdType id;
  std::vector<uint8_t> data;
};

class CanbusSocket {
 public:
  CanbusSocket() : socket_fd_(-1) {}

  void initialize(const std::string &interface_name) {
    socket_fd_ = ::socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (socket_fd_ < 0) {
      throw std::runtime_error("Failed to open CAN socket");
    }

    struct ifreq ifr;
    std::strncpy(ifr.ifr_name, interface_name.c_str(), IFNAMSIZ - 1);
    ifr.ifr_name[IFNAMSIZ - 1] = '\0';

    const auto ioctl_status = ::ioctl(socket_fd_, SIOCGIFINDEX, &ifr);

    if (ioctl_status < 0) {
      ROS_ERROR_STREAM("Failed to get CAN interface index: " << interface_name);
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
          "Failed to bind CAN socket on interface: " << interface_name);
      throw std::runtime_error("Failed to bind CAN socket");
    }

    // Set socket to non-blocking mode
    const auto flags = fcntl(socket_fd_, F_GETFL, 0);
    if (flags == -1 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) == -1) {
      close();
      throw std::runtime_error("Failed to set CAN socket to non-blocking mode");
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

  std::optional<CanMessage> handle() {
    if (!is_connected()) {
      ROS_ERROR_STREAM("Cannot receive CAN frame: (not connected)");
      return std::nullopt;
    }

    struct can_frame frame = {};
    {
      std::scoped_lock lock{socket_mutex_};
      const auto bytes_read = ::read(socket_fd_, &frame, sizeof(frame));

      if (bytes_read < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          // No data available, this is expected in non-blocking mode
          return std::nullopt;
        } else {
          throw std::runtime_error("Failed to read from CAN socket");
        }
      }

      if (bytes_read != sizeof(frame)) {
        throw std::runtime_error("Incomplete CAN frame received");
      }
    }

    CanMessage message;
    message.id = frame.can_id & CAN_EFF_MASK;
    message.data.resize(frame.can_dlc);
    std::memcpy(message.data.data(), frame.data, frame.can_dlc);
    return message;
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

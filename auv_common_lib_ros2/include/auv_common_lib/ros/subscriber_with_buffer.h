#pragma once
#include <mutex>
#include <memory>
#include <functional>

#include "rclcpp/rclcpp.hpp"

namespace auv {
namespace common {
namespace ros {

template <typename MessageT>
class SubscriberWithBuffer {
  using SharedPtr = typename MessageT::SharedPtr;
  using CallbackT = std::function<void(const SharedPtr&)>;

 public:
  SubscriberWithBuffer(rclcpp::Node::SharedPtr node) : node_{node} {}

  template <typename MessageCallbackT, typename T>
  void subscribe(const std::string& topic, uint32_t queue_size,
                 MessageCallbackT&& callback, T* obj) {
    subscribe(topic, queue_size,
              std::bind(callback, obj, std::placeholders::_1));
  }

  void subscribe(const std::string& topic, uint32_t queue_size,
                 CallbackT&& callback) {
    message_callback_ = callback;

    auto qos = rclcpp::QoS(rclcpp::KeepLast(queue_size));
    subscriber_ = node_->create_subscription<MessageT>(
        topic, qos,
        [this](const SharedPtr msg) { this->message_callback(msg); });
  }

  void subscribe(const std::string& topic, uint32_t queue_size) {
    subscribe(topic, queue_size, nullptr);
  }

  const MessageT& get_message() {
    auto lock = std::scoped_lock{mutex_};
    return message_;
  }

 protected:
  void message_callback(const SharedPtr& msg) {
    auto lock = std::scoped_lock{mutex_};

    message_ = *msg;

    if (message_callback_) {
      message_callback_(msg);
    }
  }

  rclcpp::Node::SharedPtr node_;
  typename rclcpp::Subscription<MessageT>::SharedPtr subscriber_;
  MessageT message_;
  CallbackT message_callback_;
  std::mutex mutex_;
};

}  // namespace ros
}  // namespace common
}  // namespace auv

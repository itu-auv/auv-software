#pragma once
#include <mutex>

#include "ros/ros.h"

namespace auv {
namespace common {
namespace ros {
//

template <typename MessageT>
class SubscriberWithBuffer {
  using ConstPtr = typename MessageT::ConstPtr;
  using CallbackT = std::function<void(const ConstPtr&)>;

 public:
  SubscriberWithBuffer(const ::ros::NodeHandle& nh) : nh_{nh} {}

  template <typename MessageCallbackT, typename T>
  void subscribe(const std::string& topic, uint32_t queue_size,
                 MessageCallbackT&& callback, T* obj) {
    subscribe(topic, queue_size,
              std::bind(callback, obj, std::placeholders::_1));
  }

  void subscribe(const std::string& topic, uint32_t queue_size,
                 CallbackT&& callback) {
    message_callback_ = callback;

    subscriber_ =
        nh_.subscribe(topic, queue_size,
                      &SubscriberWithBuffer<MessageT>::message_callback, this);
  }

  void subscribe(const std::string& topic, uint32_t queue_size) {
    subscribe(topic, queue_size, nullptr);
  }

  const MessageT& get_message() {
    auto lock = std::scoped_lock{mutex_};
    return message_;
  }

 protected:
  void message_callback(const ConstPtr& msg) {
    auto lock = std::scoped_lock{mutex_};

    message_ = *msg;

    if (message_callback_) {
      message_callback_(msg);
    }
  }

  ::ros::NodeHandle nh_;
  ::ros::Subscriber subscriber_;
  MessageT message_;
  CallbackT message_callback_;
  std::mutex mutex_;
};

}  // namespace ros
}  // namespace common
}  // namespace auv

#pragma once
#include "auv_common_lib/ros/subscriber_with_buffer.h"
#include "ros/ros.h"

namespace auv {
namespace common {
namespace ros {

template <typename MessageT>
class SubscriberWithTimeout : private SubscriberWithBuffer<MessageT> {
  using ConstPtr = typename MessageT::ConstPtr;
  using CallbackT = std::function<void(const ConstPtr&)>;
  using TimeoutCallbackT = std::function<void()>;
  using Base = SubscriberWithBuffer<MessageT>;

 public:
  SubscriberWithTimeout(const ::ros::NodeHandle& nh) : Base{nh} {}

  template <typename MessageCallbackT, typename TimeoutCallbackT, typename T1,
            typename T2>
  void subscribe(const std::string& topic, uint32_t queue_size,
                 MessageCallbackT&& callback, T1* obj1,
                 TimeoutCallbackT&& _timeout_callback, T2* obj2,
                 const ::ros::Duration timeout) {
    subscribe(topic, queue_size,
              std::bind(callback, obj1, std::placeholders::_1),
              std::bind(_timeout_callback, obj2), timeout);
  }

  void subscribe(const std::string& topic, uint32_t queue_size,
                 CallbackT&& callback, TimeoutCallbackT&& timeout_callback,
                 const ::ros::Duration timeout) {
    message_callback_ = callback;
    timeout_callback_ = timeout_callback;

    Base::subscribe(topic, queue_size,
                    &SubscriberWithTimeout<MessageT>::message_callback, this);

    timeout_timer_ = this->nh_.createTimer(
        timeout, &SubscriberWithTimeout<MessageT>::timeout_callback, this, true,
        false);
  }

  void subscribe(const std::string& topic, uint32_t queue_size,
                 const ::ros::Duration timeout) {
    subscribe(topic, queue_size, nullptr, nullptr, timeout);
  }

  const MessageT& get_message() {
    if (is_timeouted() && default_message_.has_value()) {
      return *default_message_;
    }

    return Base::get_message();
  }

  bool is_timeouted() const { return timeouted_; }

  void set_default_message(const MessageT& message) {
    default_message_ = message;
  }

 private:
  void message_callback(const ConstPtr& msg) {
    timeout_timer_.stop();
    timeout_timer_.start();
    timeouted_ = false;
    if (message_callback_) {
      message_callback_(msg);
    }
  }

  void timeout_callback(const ::ros::TimerEvent& /* event */) {
    timeouted_ = true;
    if (timeout_callback_) {
      timeout_callback_();
    }
  }

  ::ros::Timer timeout_timer_;
  ::ros::Duration timeout_;

  std::optional<MessageT> default_message_;
  CallbackT message_callback_;
  bool timeouted_{false};
  TimeoutCallbackT timeout_callback_;
};

//
}  // namespace ros
}  // namespace common
}  // namespace auv

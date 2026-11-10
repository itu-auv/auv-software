// BSD 3-Clause License
//
// Copyright (c) 2021, ITU AUV Team, Sencer Yazici
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <auv_cam/auv_camera_ros.h>

#include <boost/array.hpp>
#include <boost/range/algorithm.hpp>

namespace auv_cam {
USBGStreamerCameraROS::USBGStreamerCameraROS(const rclcpp::Node::SharedPtr &nh)
    : nh_(nh), it_(nh_) {
  width_ = nh_->declare_parameter<int>("width", 1920);
  height_ = nh_->declare_parameter<int>("height", 1080);
  fps_ = nh_->declare_parameter<int>("fps", 30);
  device_ = nh_->declare_parameter<std::string>("device", "/dev/video0");
  verbose_ = nh_->declare_parameter<bool>("verbose", false);
  pub_camera_info_ = nh_->declare_parameter<bool>("pub_camera_info", false);

  if (pub_camera_info_) {
    bool success = true;
    const auto get_vector = [this, &success](const std::string &name) {
      const auto value = nh_->declare_parameter<std::vector<double>>(
          name, std::vector<double>{});
      success &= !value.empty();
      return value;
    };
    cam_info_.d = get_vector("distortion_coefficients.data");

    std::vector<double> K = get_vector("camera_matrix.data");
    boost::range::copy(K, cam_info_.k.begin());

    std::vector<double> R = get_vector("rectification_matrix.data");
    boost::range::copy(R, cam_info_.r.begin());

    std::vector<double> P = get_vector("projection_matrix.data");
    boost::range::copy(P, cam_info_.p.begin());

    cam_info_.distortion_model =
        nh_->declare_parameter<std::string>("distortion_model", "");
    int width, height;
    width = nh_->declare_parameter<int>("image_width", 0);
    height = nh_->declare_parameter<int>("image_height", 0);

    cam_info_.width = static_cast<uint32_t>(width);
    cam_info_.height = static_cast<uint32_t>(height);
  }

  if (verbose_) {
    RCLCPP_INFO_STREAM(nh_->get_logger(), "Opening GST Cam");
    RCLCPP_INFO_STREAM(nh_->get_logger(), "device:" << device_);
    RCLCPP_INFO_STREAM(nh_->get_logger(), "width:" << width_);
    RCLCPP_INFO_STREAM(nh_->get_logger(), "height:" << height_);
    RCLCPP_INFO_STREAM(nh_->get_logger(), "fps:" << fps_);
  }

  std::stringstream ss;
  ss << "v4l2src device=" << device_
     << " ! image/jpeg, format=MJPG, framerate=" << fps_
     << "/1, width=" << width_ << ", height=" << height_
     << " ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, "
        "format=(string)BGRx"
     << " ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1";
  const std::string default_gst_device = ss.str();
  if (verbose_) {
    RCLCPP_INFO_STREAM(nh_->get_logger(), "gstreamer pipeline:" << default_gst_device);
  }

  // open device
  capture_ = cv::VideoCapture(default_gst_device, cv::CAP_GSTREAMER);

  if (!capture_.isOpened()) {
    // do not proceed any further
    RCLCPP_ERROR_STREAM(nh_->get_logger(), "camera is not opened.");
    std::runtime_error("camera is not opened.");
  }

  if (pub_camera_info_) {
    cam_pub_ = it_.advertiseCamera("~/image_raw", 1);
  } else {
    image_pub_ = it_.advertise("~/image_raw", 1);
  }
  RCLCPP_INFO_STREAM(nh_->get_logger(), "started camera");
}

void USBGStreamerCameraROS::start() {
  sensor_msgs::msg::Image img_msg;
  while (rclcpp::ok()) {
    cv::Mat frame;
    capture_ >> frame;

    if (frame.empty()) {
      RCLCPP_ERROR_STREAM(nh_->get_logger(), "capture frame is empty.");
      break;
    }

    std_msgs::msg::Header header;
    header.stamp = nh_->now();

    const auto img_bridge =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame);
    img_bridge.toImageMsg(img_msg);

    if (pub_camera_info_) {
      cam_pub_.publish(img_msg, cam_info_, nh_->now());
    } else {
      image_pub_.publish(img_msg);
    }
    rclcpp::spin_some(nh_);
  }
}
}  // namespace auv_cam

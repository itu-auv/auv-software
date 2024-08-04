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
USBGStreamerCameraROS::USBGStreamerCameraROS(const ros::NodeHandle &nh)
    : nh_(nh), it_(nh_) {
  width_ = nh_.param<int>("width", 1920);
  height_ = nh_.param<int>("height", 1080);
  fps_ = nh_.param<int>("fps", 30);
  device_ = nh_.param<std::string>("device", "/dev/video0");
  verbose_ = nh_.param<bool>("verbose", false);
  pub_camera_info_ = nh_.param<bool>("pub_camera_info", false);

  if (pub_camera_info_) {
    bool success = true;
    success &= nh_.getParam("distortion_coefficients/data", cam_info_.D);

    std::vector<double> K;
    success &= nh_.getParam("camera_matrix/data", K);
    boost::range::copy(K, cam_info_.K.begin());

    std::vector<double> R;
    success &= nh_.getParam("rectification_matrix/data", R);
    boost::range::copy(R, cam_info_.R.begin());

    std::vector<double> P;
    success &= nh_.getParam("projection_matrix/data", P);
    boost::range::copy(P, cam_info_.P.begin());

    success &= nh_.getParam("distortion_model", cam_info_.distortion_model);
    int width, height;
    success &= nh_.getParam("image_width", width);
    success &= nh_.getParam("image_height", height);

    cam_info_.width = static_cast<uint32_t>(width);
    cam_info_.height = static_cast<uint32_t>(height);
  }

  if (verbose_) {
    ROS_INFO_STREAM("Opening GST Cam");
    ROS_INFO_STREAM("device:" << device_);
    ROS_INFO_STREAM("width:" << width_);
    ROS_INFO_STREAM("height:" << height_);
    ROS_INFO_STREAM("fps:" << fps_);
  }

std::stringstream ss;
ss << "v4l2src device=" << device_
   << " ! image/jpeg, format=MJPG, framerate=" << fps_ << "/1, width=" << width_ << ", height=" << height_
   << " ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw, format=(string)BGRx"
   << " ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1";
  const std::string default_gst_device = ss.str();
  if (verbose_) {
    ROS_INFO_STREAM("gstreamer pipeline:" << default_gst_device);
  }

  // open device
  capture_ = cv::VideoCapture(default_gst_device, cv::CAP_GSTREAMER);

  if (!capture_.isOpened()) {
    // do not proceed any further
    ROS_ERROR_STREAM("camera is not opened.");
    std::runtime_error("camera is not opened.");
  }

  if (pub_camera_info_) {
    cam_pub_ = it_.advertiseCamera("image_raw", 1);
  } else {
    image_pub_ = it_.advertise("image_raw", 1);
  }
  ROS_INFO_STREAM("started camera");
}

void USBGStreamerCameraROS::start() {
  sensor_msgs::Image img_msg;
  while (ros::ok()) {
    cv::Mat frame;
    capture_ >> frame;

    if (frame.empty()) {
      ROS_ERROR_STREAM("capture frame is empty.");
      break;
    }

    std_msgs::Header header;
    header.stamp = ros::Time::now();

    const auto img_bridge =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame);
    img_bridge.toImageMsg(img_msg);

    if (pub_camera_info_) {
      cam_pub_.publish(img_msg, cam_info_, ros::Time::now());
    } else {
      image_pub_.publish(img_msg);
    }
    ros::spinOnce();
  }
}
}  // namespace auv_cam

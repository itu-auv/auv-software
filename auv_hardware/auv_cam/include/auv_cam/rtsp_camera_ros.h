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

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include "opencv2/opencv.hpp"
#include "ros/ros.h"

namespace auv_cam {

/**
 * @brief RTSP IP Camera ROS node with H265 support and reconnection capability
 */
class RTSPCameraROS {
 private:
  ros::NodeHandle nh_;
  cv::VideoCapture capture_;
  
  // Camera parameters
  int height_;
  int width_;
  int fps_;
  bool verbose_;
  
  // RTSP parameters
  std::string rtsp_url_;
  int latency_;
  std::string buffer_mode_;
  bool drop_on_latency_;
  
  // Reconnection parameters
  int reconnect_interval_sec_;
  int max_consecutive_failures_;
  
  // Image transport
  image_transport::ImageTransport it_;
  image_transport::Publisher image_pub_;
  image_transport::CameraPublisher cam_pub_;
  bool pub_camera_info_;
  sensor_msgs::CameraInfo cam_info_;
  
  // Connection state
  bool is_connected_;
  int consecutive_failures_;

  /**
   * @brief Build the GStreamer pipeline string for RTSP H265
   * @return GStreamer pipeline string
   */
  std::string buildGStreamerPipeline() const;

  /**
   * @brief Attempt to open the RTSP stream
   * @return true if connection successful, false otherwise
   */
  bool openStream();

  /**
   * @brief Close the current stream connection
   */
  void closeStream();

 public:
  /**
   * @brief Construct a new RTSPCameraROS object
   * @param nh Node handle
   */
  RTSPCameraROS(const ros::NodeHandle &nh);

  /**
   * @brief Destructor
   */
  ~RTSPCameraROS();

  /**
   * @brief Start polling frames and publish with reconnection support
   */
  void start();
};

}  // namespace auv_cam

// BSD 3-Clause License
//
// Copyright (c) 2025, ITU AUV Team, Faruk Mimarlar
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

#include <auv_cam/rtsp_camera_ros.h>

#include <boost/array.hpp>
#include <boost/range/algorithm.hpp>
#include <chrono>
#include <thread>

namespace auv_cam {

RTSPCameraROS::RTSPCameraROS(const ros::NodeHandle &nh)
    : nh_(nh), 
      it_(nh_),
      is_connected_(false),
      consecutive_failures_(0) {
  
  // Video parameters
  width_ = nh_.param<int>("width", 1920);
  height_ = nh_.param<int>("height", 1080);
  fps_ = nh_.param<int>("fps", 30);
  verbose_ = nh_.param<bool>("verbose", false);
  pub_camera_info_ = nh_.param<bool>("pub_camera_info", false);

  // RTSP parameters
  rtsp_url_ = nh_.param<std::string>("rtsp_url", "rtsp://192.168.1.65/554");
  latency_ = nh_.param<int>("latency", 0);
  buffer_mode_ = nh_.param<std::string>("buffer_mode", "auto");
  drop_on_latency_ = nh_.param<bool>("drop_on_latency", true);
  
  // Reconnection parameters
  reconnect_interval_sec_ = nh_.param<int>("reconnect_interval_sec", 5);
  max_consecutive_failures_ = nh_.param<int>("max_consecutive_failures", 100);

  // Load camera info if needed
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

    if (!success) {
      ROS_WARN_STREAM("Failed to load complete camera info parameters");
    }
  }

  if (verbose_) {
    ROS_INFO_STREAM("Opening RTSP Camera");
    ROS_INFO_STREAM("rtsp_url: " << rtsp_url_);
    ROS_INFO_STREAM("latency: " << latency_);
    ROS_INFO_STREAM("buffer_mode: " << buffer_mode_);
    ROS_INFO_STREAM("drop_on_latency: " << (drop_on_latency_ ? "true" : "false"));
    ROS_INFO_STREAM("width: " << width_);
    ROS_INFO_STREAM("height: " << height_);
    ROS_INFO_STREAM("fps: " << fps_);
    ROS_INFO_STREAM("reconnect_interval_sec: " << reconnect_interval_sec_);
    ROS_INFO_STREAM("max_consecutive_failures: " << max_consecutive_failures_);
  }

  // Setup publishers
  if (pub_camera_info_) {
    cam_pub_ = it_.advertiseCamera("image_raw", 1);
  } else {
    image_pub_ = it_.advertise("image_raw", 1);
  }

  // Connection will be attempted in start() after a small delay
  // This ensures ROS is fully initialized before GStreamer starts
  ROS_INFO_STREAM("RTSP Camera initialized. Connection will start in start() loop.");
}

RTSPCameraROS::~RTSPCameraROS() {
  closeStream();
}

std::string RTSPCameraROS::buildGStreamerPipeline() const {
  std::stringstream ss;
  
  // RTSP source - matches the user's working gst-launch command
  ss << "rtspsrc location=\"" << rtsp_url_ << "\""
     << " latency=" << latency_
     << " ! ";
  
  // Explicit H265 decode chain (as per user's working command)
  ss << "rtph265depay ! h265parse ! avdec_h265 ! ";
  
  // Video conversion to BGR for OpenCV
  ss << "videoconvert ! video/x-raw,format=(string)BGR ! ";
  
  // Appsink configured for OpenCV compatibility
  // name=appsink0 is required for OpenCV to find the sink
  // emit-signals=true allows OpenCV to pull frames
  ss << "appsink name=appsink0 drop=true max-buffers=1 sync=false emit-signals=true";
  
  return ss.str();
}

bool RTSPCameraROS::openStream() {
  const std::string pipeline = buildGStreamerPipeline();
  
  if (verbose_) {
    ROS_INFO_STREAM("GStreamer pipeline: " << pipeline);
  }

  capture_ = cv::VideoCapture(pipeline, cv::CAP_GSTREAMER);

  if (!capture_.isOpened()) {
    is_connected_ = false;
    ROS_ERROR_STREAM("Failed to open RTSP stream: " << rtsp_url_);
    return false;
  }
  
  // Verify connection by actually reading a frame
  // This catches cases where isOpened() returns true but pipeline isn't working
  ROS_INFO_STREAM("Verifying stream connection...");
  cv::Mat test_frame;
  int verify_attempts = 0;
  const int max_verify_attempts = 30;  // Try for ~3 seconds
  
  while (verify_attempts < max_verify_attempts) {
    if (capture_.read(test_frame) && !test_frame.empty()) {
      is_connected_ = true;
      consecutive_failures_ = 0;
      ROS_INFO_STREAM("Successfully connected to RTSP stream: " << rtsp_url_);
      ROS_INFO_STREAM("Frame size: " << test_frame.cols << "x" << test_frame.rows);
      return true;
    }
    verify_attempts++;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  
  // If we couldn't read a frame, connection failed
  ROS_ERROR_STREAM("Stream opened but failed to read frames from: " << rtsp_url_);
  capture_.release();
  is_connected_ = false;
  return false;
}

void RTSPCameraROS::closeStream() {
  if (capture_.isOpened()) {
    capture_.release();
  }
  is_connected_ = false;
}

void RTSPCameraROS::start() {
  sensor_msgs::Image img_msg;
  
  // Small delay before first connection to ensure ROS is fully ready
  ROS_INFO_STREAM("Waiting 1 second before initial connection...");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  
  while (ros::ok()) {
    // If not connected, try to reconnect
    if (!is_connected_) {
      ROS_INFO_STREAM("Attempting connection to: " << rtsp_url_);
      
      if (openStream()) {
        ROS_INFO_STREAM("Reconnection successful!");
      } else {
        ROS_WARN_STREAM("Reconnection failed. Retrying in " 
                        << reconnect_interval_sec_ << " seconds...");
        
        // Wait before retry, but check ros::ok() periodically
        for (int i = 0; i < reconnect_interval_sec_ && ros::ok(); ++i) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          ros::spinOnce();
        }
        continue;
      }
    }

    // Try to read frame
    cv::Mat frame;
    bool read_success = capture_.read(frame);

    if (!read_success || frame.empty()) {
      consecutive_failures_++;
      ROS_WARN_STREAM_THROTTLE(5, "Failed to read frame. Consecutive failures: " 
                               << consecutive_failures_);
      
      // If too many failures, consider connection lost
      if (consecutive_failures_ >= max_consecutive_failures_) {
        ROS_ERROR_STREAM("Max consecutive failures reached (" 
                         << max_consecutive_failures_ 
                         << "). Closing stream and will retry...");
        closeStream();
        
        // Wait before reconnection attempt
        std::this_thread::sleep_for(std::chrono::seconds(reconnect_interval_sec_));
      }
      
      // Short delay to avoid busy loop on continuous failures
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      ros::spinOnce();
      continue;
    }

    // Successful frame read - reset failure counter
    consecutive_failures_ = 0;

    // Build and publish message
    std_msgs::Header header;
    header.stamp = ros::Time::now();

    const auto img_bridge =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame);
    img_bridge.toImageMsg(img_msg);

    if (pub_camera_info_) {
      cam_info_.header = header;
      cam_pub_.publish(img_msg, cam_info_, ros::Time::now());
    } else {
      image_pub_.publish(img_msg);
    }
    
    ros::spinOnce();
  }
  
  // Cleanup on exit
  closeStream();
  ROS_INFO_STREAM("RTSP Camera node shutting down.");
}

}  // namespace auv_cam

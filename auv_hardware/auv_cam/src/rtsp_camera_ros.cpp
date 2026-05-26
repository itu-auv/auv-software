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

namespace auv_cam {

RTSPCameraROS::RTSPCameraROS(const ros::NodeHandle &nh)
    : nh_(nh), pipeline_(nullptr), appsink_(nullptr), it_(nh_) {
  // GStreamer must be initialized exactly once per process. Safe to call
  // multiple times; subsequent calls are no-ops.
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }

  width_ = nh_.param<int>("width", 640);
  height_ = nh_.param<int>("height", 480);
  fps_ = nh_.param<int>("fps", 30);
  verbose_ = nh_.param<bool>("verbose", false);
  pub_camera_info_ = nh_.param<bool>("pub_camera_info", false);

  // RTSP parameters
  rtsp_url_ = nh_.param<std::string>("rtsp_url", "rtsp://192.168.1.65:554/");
  latency_ = nh_.param<int>("latency", 200);

  // Optional manual offset added to the PTS-derived stamp. Default 0 because
  // the appsink path computes per-frame latency from the GStreamer pipeline
  // clock. Set non-zero only if a residual systematic bias is observed.
  stamp_offset_ = nh_.param<double>("stamp_offset", 0.0);

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
  }

  if (verbose_) {
    ROS_INFO_STREAM("Opening RTSP Camera");
    ROS_INFO_STREAM("rtsp_url: " << rtsp_url_);
    ROS_INFO_STREAM("latency: " << latency_);
    ROS_INFO_STREAM("width: " << width_);
    ROS_INFO_STREAM("height: " << height_);
    ROS_INFO_STREAM("fps: " << fps_);
  }

  // Build GStreamer pipeline. Uses Jetson HW decoder (nvv4l2decoder) and
  // converter (nvvidconv). The appsink is named so we can fetch buffer PTS
  // alongside the pixel data; sync=false lets us pull frames as fast as the
  // decoder can deliver and max-buffers=1+drop=true keep us on the freshest
  // frame instead of falling further behind.
  std::stringstream ss;
  ss << "rtspsrc location=\"" << rtsp_url_ << "\""
     << " latency=" << latency_ << " protocols=tcp"
     << " ! rtph265depay ! h265parse"
     << " ! nvv4l2decoder"
     << " ! nvvidconv ! video/x-raw, format=(string)BGRx"
     << " ! videoconvert ! video/x-raw, format=(string)BGR"
     << " ! appsink name=sink sync=false max-buffers=1 drop=true";

  const std::string gst_pipeline = ss.str();

  if (verbose_) {
    ROS_INFO_STREAM("GStreamer pipeline: " << gst_pipeline);
  }

  GError *err = nullptr;
  pipeline_ = gst_parse_launch(gst_pipeline.c_str(), &err);
  if (err != nullptr) {
    const std::string msg = err->message;
    g_error_free(err);
    if (pipeline_ != nullptr) {
      gst_object_unref(pipeline_);
      pipeline_ = nullptr;
    }
    ROS_ERROR_STREAM("gst_parse_launch failed: " << msg);
    throw std::runtime_error("gst_parse_launch failed: " + msg);
  }

  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (sink == nullptr) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
    ROS_ERROR_STREAM("appsink element 'sink' not found in pipeline");
    throw std::runtime_error("appsink not found");
  }
  appsink_ = GST_APP_SINK(sink);  // takes ownership of the ref

  const GstStateChangeReturn ret =
      gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    ROS_ERROR_STREAM("Failed to set GStreamer pipeline to PLAYING");
    throw std::runtime_error("GStreamer pipeline failed to start");
  }

  if (pub_camera_info_) {
    cam_pub_ = it_.advertiseCamera("image_raw", 1);
  } else {
    image_pub_ = it_.advertise("image_raw", 1);
  }

  ROS_INFO_STREAM("Started RTSP camera: " << rtsp_url_);
}

RTSPCameraROS::~RTSPCameraROS() {
  if (pipeline_ != nullptr) {
    gst_element_set_state(pipeline_, GST_STATE_NULL);
  }
  if (appsink_ != nullptr) {
    gst_object_unref(appsink_);
    appsink_ = nullptr;
  }
  if (pipeline_ != nullptr) {
    gst_object_unref(pipeline_);
    pipeline_ = nullptr;
  }
}

void RTSPCameraROS::start() {
  sensor_msgs::Image img_msg;

  while (ros::ok()) {
    // Block up to 1 s for the next decoded frame so we react quickly to
    // shutdown and to RTSP stalls.
    GstSample *sample =
        gst_app_sink_try_pull_sample(appsink_, GST_SECOND);
    if (sample == nullptr) {
      if (gst_app_sink_is_eos(appsink_)) {
        ROS_ERROR_STREAM("appsink reached EOS, stopping");
        break;
      }
      ROS_WARN_THROTTLE(2.0, "No RTSP sample within 1 s");
      continue;
    }

    GstBuffer *buf = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    if (buf == nullptr || caps == nullptr) {
      gst_sample_unref(sample);
      continue;
    }

    // Read frame dimensions from caps so we never trust a launch arg that
    // disagrees with what the RTSP source is actually delivering.
    GstStructure *cap_struct = gst_caps_get_structure(caps, 0);
    int frame_w = 0, frame_h = 0;
    gst_structure_get_int(cap_struct, "width", &frame_w);
    gst_structure_get_int(cap_struct, "height", &frame_h);
    if (frame_w <= 0 || frame_h <= 0) {
      gst_sample_unref(sample);
      continue;
    }

    // Compute the buffer's age in the GStreamer pipeline, which is what we
    // subtract from ros::Time::now() to recover the capture time.
    //
    //   buffer_age_ns = pipeline_running_time_now - GST_BUFFER_PTS(buf)
    //   capture_stamp = ros::Time::now() - buffer_age_ns
    //
    // PTS is set by rtspsrc from the RTP timestamps emitted by the camera,
    // so this captures network + jitter buffer + decode latency in one shot
    // without requiring a hand-tuned offset.
    double buffer_age_s = 0.0;
    GstClockTime pts = GST_BUFFER_PTS(buf);
    GstClock *clock = gst_element_get_clock(pipeline_);
    if (clock != nullptr && GST_CLOCK_TIME_IS_VALID(pts)) {
      const GstClockTime base_time = gst_element_get_base_time(pipeline_);
      const GstClockTime now_clock = gst_clock_get_time(clock);
      if (GST_CLOCK_TIME_IS_VALID(now_clock) &&
          GST_CLOCK_TIME_IS_VALID(base_time) && now_clock >= base_time) {
        const GstClockTime running_now = now_clock - base_time;
        if (running_now >= pts) {
          buffer_age_s = static_cast<double>(running_now - pts) / 1e9;
        }
      }
      gst_object_unref(clock);
    }

    const ros::Time stamp =
        ros::Time::now() - ros::Duration(buffer_age_s + stamp_offset_);

    if (verbose_) {
      ROS_INFO_THROTTLE(
          2.0,
          "RTSP frame %dx%d, pipeline latency: %.1f ms (manual offset: %.1f ms)",
          frame_w, frame_h, buffer_age_s * 1000.0, stamp_offset_ * 1000.0);
    }

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
      gst_sample_unref(sample);
      ROS_ERROR_STREAM("Failed to map GStreamer buffer");
      continue;
    }

    // Wrap the GStreamer buffer memory; cv_bridge will copy into the ROS msg.
    cv::Mat frame(frame_h, frame_w, CV_8UC3, map.data);

    std_msgs::Header header;
    header.stamp = stamp;

    const auto img_bridge =
        cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, frame);
    img_bridge.toImageMsg(img_msg);

    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);

    if (pub_camera_info_) {
      cam_pub_.publish(img_msg, cam_info_, stamp);
    } else {
      image_pub_.publish(img_msg);
    }

    ros::spinOnce();
  }
}

}  // namespace auv_cam

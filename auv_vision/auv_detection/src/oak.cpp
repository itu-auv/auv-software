#include <memory>
#include <string>
#include <vector>

#include "depthai/depthai.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "vision_msgs/Detection2D.h"
#include "vision_msgs/Detection2DArray.h"
#include "vision_msgs/ObjectHypothesisWithPose.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "oak_yolo_detection_cpp");
  ros::NodeHandle nh("~");
  ros::Publisher detections_pub =
      nh.advertise<vision_msgs::Detection2DArray>("/oak/yolo/detections", 1);
  ros::Publisher image_pub =
      nh.advertise<sensor_msgs::Image>("/oak/yolo/image_raw", 1);

  // Pipeline setup
  dai::Pipeline pipeline;

  auto cam = pipeline.create<dai::node::ColorCamera>();
  cam->setPreviewSize(416, 416);  // Model input size
  cam->setInterleaved(false);
  cam->setColorOrder(dai::ColorCameraProperties::ColorOrder::BGR);
  cam->setFps(15);

  auto manip = pipeline.create<dai::node::ImageManip>();
  manip->initialConfig.setResize(416, 416);
  manip->initialConfig.setKeepAspectRatio(false);
  cam->preview.link(manip->inputImage);

  auto nn = pipeline.create<dai::node::NeuralNetwork>();
  std::string blob_path;
  nh.getParam("model_path",
              blob_path);  // launch dosyasından parametre geçebilirsin
  nn->setBlobPath(blob_path);
  manip->out.link(nn->input);

  auto xout_nn = pipeline.create<dai::node::XLinkOut>();
  xout_nn->setStreamName("nn");
  nn->out.link(xout_nn->input);

  auto xout_img = pipeline.create<dai::node::XLinkOut>();
  xout_img->setStreamName("img");
  cam->preview.link(xout_img->input);

  // Device ve queue’ları aç
  dai::Device device(pipeline);
  auto nnQ = device.getOutputQueue("nn", 4, false);
  auto imgQ = device.getOutputQueue("img", 1, false);

  ros::Rate rate(10);
  while (ros::ok()) {
    auto in_nn = nnQ->tryGet<dai::NNData>();
    auto in_img = imgQ->tryGet<dai::ImgFrame>();

    if (in_nn && in_img) {
      cv::Mat image = in_img->getCvFrame();
      vision_msgs::Detection2DArray det_array_msg;
      det_array_msg.header.stamp = ros::Time::now();
      det_array_msg.header.frame_id = "oak_rgb_camera_frame";

      // Output layer ismi modeline göre değişebilir!
      std::vector<float> raw_output = in_nn->getLayerFp16("output0");
      size_t det_len = 84;  // Modelinin output'u ile aynı olmalı

      if (!raw_output.empty() && raw_output.size() % det_len == 0) {
        size_t num_dets = raw_output.size() / det_len;
        for (size_t i = 0; i < num_dets; ++i) {
          const float* det = raw_output.data() + i * det_len;
          float objectness = 1.0f / (1.0f + std::exp(-det[4]));
          std::vector<float> class_probs(det + 5, det + det_len);
          float max_prob =
              *std::max_element(class_probs.begin(), class_probs.end());
          int class_id = std::distance(
              class_probs.begin(),
              std::max_element(class_probs.begin(), class_probs.end()));
          float conf = objectness * (1.0f / (1.0f + std::exp(-max_prob)));

          if (conf > 0.4) {
            float x_center = det[0], y_center = det[1], width = det[2],
                  height = det[3];
            int x1 = static_cast<int>(x_center - width / 2);
            int y1 = static_cast<int>(y_center - height / 2);
            int x2 = static_cast<int>(x_center + width / 2);
            int y2 = static_cast<int>(y_center + height / 2);

            // ROS Detection2D mesajı oluştur
            vision_msgs::Detection2D detection_msg;
            detection_msg.bbox.center.x = x_center;
            detection_msg.bbox.center.y = y_center;
            detection_msg.bbox.size_x = width;
            detection_msg.bbox.size_y = height;
            vision_msgs::ObjectHypothesisWithPose hypo;
            hypo.id = class_id;
            hypo.score = conf;
            detection_msg.results.push_back(hypo);
            det_array_msg.detections.push_back(detection_msg);

            // Görüntüye çiz
            cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 2);
          }
        }
      }
      // Görüntü publish (cv_bridge kullanarak ROS Image mesajına çevir)
      sensor_msgs::Image img_msg;
      img_msg.header = det_array_msg.header;
      img_msg.height = image.rows;
      img_msg.width = image.cols;
      img_msg.encoding = "bgr8";
      img_msg.step = image.step;
      size_t size = img_msg.step * img_msg.height;
      img_msg.data.resize(size);
      memcpy(&img_msg.data[0], image.data, size);

      detections_pub.publish(det_array_msg);
      image_pub.publish(img_msg);
    }
    ros::spinOnce();
    rate.sleep();
  }
  return 0;
}

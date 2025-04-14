#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnythingNode:
    def __init__(self):
        rospy.init_node('depth_anything_node', anonymous=True)

        # ROS Image subscriber
        self.image_sub = rospy.Subscriber("/taluy/cameras/cam_front/image_raw", Image, self.image_callback)

        # Depth Image publisher
        self.depth_pub = rospy.Publisher("/depth_any/camera/depth/image_raw", Image, queue_size=1)

        # CV Bridge
        self.bridge = CvBridge()

        # Depth Model (GPU desteğiyle)
        model_name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

        # CUDA kontrolü
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"CUDA kullanılabilir mi? {torch.cuda.is_available()}")

        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        ).to(self.device)

        rospy.loginfo("Depth Anything Node Başlatıldı...")

    def image_callback(self, msg):
        try:
            # ROS Image mesajını OpenCV formatına çevir
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # OpenCV görüntüsünü PIL formata çevir
            image_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Derinlik tahmini girişleri ve GPU'ya taşıma
            inputs = self.image_processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()  # GPU'dan CPU'ya al

            # Depth map'i orijinal görüntü boyutuna getir (float32)
            depth_resized = cv2.resize(depth, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_CUBIC)

            # ROS mesajına çevir
            depth_msg = self.bridge.cv2_to_imgmsg(depth_resized, encoding="32FC1")
            depth_msg.header = msg.header
            self.depth_pub.publish(depth_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatası: {e}")
        except Exception as e:
            rospy.logerr(f"Derinlik tahmini sırasında hata: {e}")

if __name__ == "__main__":
    try:
        node = DepthAnythingNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

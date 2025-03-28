#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthEstimatorNode:
    def __init__(self):
        rospy.init_node('depth_estimator_node', anonymous=True)

        # ROS Image subscriber
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)

        # Depth map publisher
        self.depth_pub = rospy.Publisher("/camera/depth_map", Image, queue_size=1)

        # CV Bridge
        self.bridge = CvBridge()

        # Depth Model
        model_name = "LiheYoung/depth-anything-small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float32)

        rospy.loginfo("Depth Estimator Node Başlatıldı...")

    def image_callback(self, msg):
        try:
            # ROS Image mesajını OpenCV formatına çevir
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # OpenCV görüntüsünü PIL formata çevir
            image_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Derinlik tahmini yap
            inputs = self.image_processor(images=image_pil, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()

            # Depth map'i orijinal görüntü boyutuna getir
            depth_resized = cv2.resize(depth, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Depth map normalizasyon & renklendirme
            depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # OpenCV ile göster
            cv2.imshow("Depth Map", depth_colored)
            cv2.waitKey(1)

            # Depth map'i ROS topic olarak yayınla
            depth_msg = self.bridge.cv2_to_imgmsg(depth_colored, encoding="bgr8")
            self.depth_pub.publish(depth_msg)

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatası: {e}")

if __name__ == "__main__":
    try:
        node = DepthEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

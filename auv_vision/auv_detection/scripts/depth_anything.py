#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tf2_ros import TransformBroadcaster    
from tf.transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped  
class DepthToPointCloudNode:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud_node', anonymous=True)

        # ROS Image subscriber
        self.image_sub = rospy.Subscriber("/taluy/cameras/cam_front/image_raw", Image, self.image_callback)

        # PointCloud publisher
        self.pc_pub = rospy.Publisher("/camera/depth_cloud", PointCloud2, queue_size=1)

        # CV Bridge
        self.bridge = CvBridge()

        # Depth Model
        model_name = "LiheYoung/depth-anything-small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float32)

        # Kamera Parametreleri (örnek değerler, senin kamerana göre ayarla!)
        self.fx = 525.0  # Odak uzaklığı (X ekseni)
        self.fy = 525.0  # Odak uzaklığı (Y ekseni)
        self.cx = 320.0  # Optik merkez (X ekseni)
        self.cy = 240.0  # Optik merkez (Y ekseni)
        self.tf_broadcaster = TransformBroadcaster()
        self.publish_tf_timer = rospy.Timer(rospy.Duration(0.1), self.publish_tf)
        rospy.loginfo("Depth to PointCloud Node Başlatıldı...")
    def publish_tf(self, event):
        """ Kamera pozisyonunu map çerçevesine bağlayan TF yayını """
        """ Kamera pozisyonunu map çerçevesine bağlayan TF yayını """
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"  # Sabit frame
        t.child_frame_id = "camera_link"  # Kameranın frame'i
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        # Euler açılarını quaternion'a çevir
        q = quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)
    # TF yayını yap
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

            # 3D Noktaları Hesapla (Point Cloud)
            points = []
            height, width = depth_resized.shape
            for v in range(height):
                for u in range(width):
                    Z = depth_resized[v, u]  # Derinlik değeri
                    if Z > 0:  # Sıfır derinlikler hatalı olabilir
                        X = (u - self.cx) * Z / self.fx
                        Y = (v - self.cy) * Z / self.fy
                        points.append([X, Y, Z])

            # PointCloud2 formatına çevir
            cloud_msg = self.create_pointcloud2_msg(points)
            self.pc_pub.publish(cloud_msg)
            print("Point Cloud gönderildi...")

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatası: {e}")

    def create_pointcloud2_msg(self, points):
        """ 3D Noktaları PointCloud2 mesajına çevirir """
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_link"  # Kameranın referans çerçevesi

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
        ]

        cloud_msg = pc2.create_cloud(header, fields, points)
        return cloud_msg

if __name__ == "__main__":
    try:
        node = DepthToPointCloudNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

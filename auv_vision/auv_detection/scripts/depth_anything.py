#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
# import sensor_msgs.point_cloud2 as pc2 # Removed
from sensor_msgs.msg import Image # Modified
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
# from tf2_ros import TransformBroadcaster # Removed
# from tf.transformations import quaternion_from_euler # Removed
# from geometry_msgs.msg import TransformStamped # Removed

class DepthAnythingNode: # Renamed class
    def __init__(self):
        rospy.init_node('depth_anything_node', anonymous=True) # Renamed node

        # ROS Image subscriber
        self.image_sub = rospy.Subscriber("/taluy/cameras/cam_front/image_raw", Image, self.image_callback) # Input topic

        # Depth Image publisher
        self.depth_pub = rospy.Publisher("/depth_any/camera/depth/image_raw", Image, queue_size=1) # Output topic

        # CV Bridge
        self.bridge = CvBridge()

        # Depth Model
        model_name = "LiheYoung/depth-anything-small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float32)

        # Kamera Parametreleri (isteğe bağlı, derinlik yorumlama için kullanılabilir)
        # self.fx = rospy.get_param('~fx', 525.0)
        # self.fy = rospy.get_param('~fy', 525.0)
        # self.cx = rospy.get_param('~cx', 320.0)
        # self.cy = rospy.get_param('~cy', 240.0)
        # self.tf_broadcaster = TransformBroadcaster() # Removed
        # self.publish_tf_timer = rospy.Timer(rospy.Duration(0.1), self.publish_tf) # Removed
        rospy.loginfo("Depth Anything Node Başlatıldı...") # Updated log message

    # def publish_tf(self, event): # Removed TF publisher method
    #     """ Kamera pozisyonunu map çerçevesine bağlayan TF yayını """
    #     t = TransformStamped()
    #     t.header.stamp = rospy.Time.now()
    #     t.header.frame_id = "map"
    #     t.child_frame_id = "camera_link"
    #     t.transform.translation.x = 0.0
    #     t.transform.translation.y = 0.0
    #     t.transform.translation.z = 0.0
    #     q = quaternion_from_euler(0, 0, 0)
    #     t.transform.rotation.x = q[0]
    #     t.transform.rotation.y = q[1]
    #     t.transform.rotation.z = q[2]
    #     t.transform.rotation.w = q[3]
    #     self.tf_broadcaster.sendTransform(t)

    def image_callback(self, msg):
        try:
            # ROS Image mesajını OpenCV formatına çevir (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # OpenCV görüntüsünü PIL formata çevir
            image_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            # Derinlik tahmini yap
            inputs = self.image_processor(images=image_pil, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()

            # Depth map'i orijinal görüntü boyutuna getir (float32)
            depth_resized = cv2.resize(depth, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Derinlik görüntüsünü ROS Image mesajına çevir (32FC1)
            # Model çıktısı genellikle normalize edilmiş float değerleridir.
            # Eğer belirli bir birime (örn. metre) ölçeklemek gerekiyorsa burada yapılabilir.
            depth_msg = self.bridge.cv2_to_imgmsg(depth_resized, encoding="32FC1")
            depth_msg.header = msg.header # Gelen mesajın header'ını kullan

            # Derinlik görüntüsünü yayınla
            self.depth_pub.publish(depth_msg)
            # rospy.loginfo("Derinlik görüntüsü yayınlandı.") # İsteğe bağlı log

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatası: {e}")
        except Exception as e:
            rospy.logerr(f"Derinlik tahmini sırasında hata: {e}")

    # def create_pointcloud2_msg(self, points): # Removed PointCloud creation method
    #     """ 3D Noktaları PointCloud2 mesajına çevirir """
    #     header = rospy.Header()
    #     header.stamp = rospy.Time.now()
    #     header.frame_id = "camera_link"
    #     fields = [
    #         PointField('x', 0, PointField.FLOAT32, 1),
    #         PointField('y', 4, PointField.FLOAT32, 1),
    #         PointField('z', 8, PointField.FLOAT32, 1),
    #     ]
    #     cloud_msg = pc2.create_cloud(header, fields, points)
    #     return cloud_msg

if __name__ == "__main__":
    try:
        node = DepthAnythingNode() # Use updated class name
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

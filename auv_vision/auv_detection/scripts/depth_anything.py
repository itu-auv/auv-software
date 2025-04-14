#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import torch
import os # Dosya sistemi işlemleri için eklendi
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
        model_name = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float32)
        # self.publish_tf_timer = rospy.Timer(rospy.Duration(0.1), self.publish_tf) # Removed
        rospy.loginfo("Depth Anything Node Başlatıldı...") # Updated log message

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
                depth = outputs.predicted_depth.squeeze().gpu().numpy()

            # Depth map'i orijinal görüntü boyutuna getir (float32)
            depth_resized = cv2.resize(depth, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Derinlik görüntüsünü ROS Image mesajına çevir (32FC1)
            # Model çıktısı genellikle normalize edilmiş float değerleridir.
            # Eğer belirli bir birime (örn. metre) ölçeklemek gerekiyorsa burada yapılabilir.
            depth_msg = self.bridge.cv2_to_imgmsg(depth_resized, encoding="32FC1")
            depth_msg.header = msg.header # Gelen mesajın header'ını kullan
            """"
            # --- Görselleştirme Başlangıcı ---
            try:
                # Derinlik haritasını görselleştirme için normalize et (0-255, uint8)
                # Not: depth_resized'in min/max değerleri her karede değişebilir.
                # Daha tutarlı bir görselleştirme için sabit bir min/max veya
                # hareketli bir ortalama kullanılabilir, ancak şimdilik basit tutalım.
                depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                # Renk paleti uygula (örneğin JET)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                # --- Renk Çubuğu (Lejant) Oluşturma ---
                min_depth = np.min(depth_resized)
                max_depth = np.max(depth_resized)
                colorbar_height = depth_colormap.shape[0]
                colorbar_width = 50 # Renk çubuğu genişliği
                colorbar = np.zeros((colorbar_height, colorbar_width, 3), dtype=np.uint8)
                for i in range(colorbar_height):
                    # Renk çubuğunu yukarıdan aşağıya doğru doldur (max'dan min'e)
                    normalized_val = 255 - int((i / colorbar_height) * 255)
                    color = cv2.applyColorMap(np.array([[normalized_val]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
                    colorbar[i, :] = color

                # Min ve Max değerlerini renk çubuğuna yaz
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(colorbar, f"{max_depth:.2f}", (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(colorbar, f"{min_depth:.2f}", (5, colorbar_height - 10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                # --- Renk Çubuğu Sonu ---

                # Derinlik haritası ve renk çubuğunu birleştir
                combined_image = np.hstack((depth_colormap, colorbar))

                # Birleştirilmiş görüntüyü göster
                cv2.imshow("Depth Colormap with Legend", combined_image)
                cv2.waitKey(1) # imshow'un döngü içinde çalışması için önemli
            except Exception as vis_e:
                rospy.logwarn_throttle(10, f"Görselleştirme hatası: {vis_e}") # Hata olursa logla ama devam et
            # --- Görselleştirme Sonu ---
            """
            # Derinlik görüntüsünü yayınla
            self.depth_pub.publish(depth_msg)
            # rospy.loginfo("Derinlik görüntüsü yayınlandı.") # İsteğe bağlı log

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatası: {e}")
        except Exception as e:
            rospy.logerr(f"Derinlik tahmini sırasında hata: {e}")
if __name__ == "__main__":
    try:
        node = DepthAnythingNode() # Use updated class name
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

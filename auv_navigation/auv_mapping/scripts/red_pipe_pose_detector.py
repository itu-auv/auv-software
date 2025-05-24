#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from cv_bridge import CvBridge
import tf.transformations as tf_trans
from std_msgs.msg import Header
import yaml


class RedPipePoseDetector:
    def __init__(self):
        rospy.init_node("red_pipe_pose_detector", anonymous=True)

        # Parameters
        self.pipe_diameter = rospy.get_param(
            "~pipe_diameter", 0.0127
        )  # 5cm varsayılan PVC boru çapı
        self.min_contour_area = rospy.get_param("~min_contour_area", 500)
        self.max_contour_area = rospy.get_param("~max_contour_area", 50000)
        self.safety_margin = rospy.get_param(
            "~safety_margin", 0.2
        )  # 20cm güvenlik mesafesi

        # HSV renk aralıkları (kırmızı için)
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # ROS setup
        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # Subscribers
        self.image_sub = rospy.Subscriber(
            "/camera/image_raw", Image, self.image_callback
        )
        self.camera_info_sub = rospy.Subscriber(
            "/camera/camera_info", CameraInfo, self.camera_info_callback
        )

        # Publishers
        self.pose_pub = rospy.Publisher("/red_pipes/poses", PoseArray, queue_size=10)
        self.debug_image_pub = rospy.Publisher(
            "/red_pipes/debug_image", Image, queue_size=1
        )

        # Debug
        self.publish_debug = rospy.get_param("~publish_debug", True)

        rospy.loginfo("Red Pipe Pose Detector initialized")
        rospy.loginfo(f"Pipe diameter: {self.pipe_diameter}m")
        rospy.loginfo(f"Safety margin: {self.safety_margin}m")

    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)

            # Focal length (piksel cinsinden)
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]

            rospy.loginfo("Camera calibration received")
            rospy.loginfo(f"Focal length: fx={self.fx:.2f}, fy={self.fy:.2f}")
            rospy.loginfo(f"Principal point: cx={self.cx:.2f}, cy={self.cy:.2f}")

    def detect_red_objects(self, image):
        # HSV'ye çevir
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Kırmızı için iki maske (renk spektrumunun iki ucunda)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Morfolojik işlemler (gürültü temizleme)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Kontürları bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, mask

    def estimate_pipe_pose(self, contour, image_shape):
        """Kontürden boru pozisyonunu ve yönünü hesapla"""
        if self.camera_matrix is None:
            return None

        # Kontür alanı kontrolü
        area = cv2.contourArea(contour)
        if area < self.min_contour_area or area > self.max_contour_area:
            return None

        # Minimum enclosing circle ile boru merkezi ve yarıçapı
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Çok küçük veya büyük daireler için filtre
        if radius < 10 or radius > 200:
            return None

        # Derinlik hesaplama (gerçek boyut / piksel boyutu * focal length)
        pixel_diameter = 2 * radius
        distance = (self.pipe_diameter * self.fx) / pixel_diameter

        # Kamera koordinat sisteminde 3D pozisyon
        # Z: derinlik (kameradan uzaklık)
        # X: sağ/sol (piksel merkez noktasından sapma)
        # Y: yukarı/aşağı (piksel merkez noktasından sapma)
        z = distance
        x = (center[0] - self.cx) * z / self.fx
        y = (center[1] - self.cy) * z / self.fy

        # Elips fit ile yön belirleme (borunun açısı)
        if len(contour) >= 5:  # Elips fit için minimum 5 nokta
            ellipse = cv2.fitEllipse(contour)
            angle = np.radians(ellipse[2])  # Açıyı radyana çevir
        else:
            angle = 0.0

        # Quaternion hesaplama (Z ekseni etrafında rotasyon)
        quaternion = tf_trans.quaternion_from_euler(0, 0, angle)

        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = Quaternion(
            x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]
        )

        return pose, center, radius, distance

    def draw_debug_info(self, image, poses_info):
        debug_image = image.copy()

        for pose, center, radius, distance in poses_info:
            # Daire çiz
            cv2.circle(debug_image, center, radius, (0, 255, 0), 2)
            cv2.circle(debug_image, center, 2, (0, 0, 255), -1)

            # Mesafe bilgisi
            text = f"Dist: {distance:.2f}m"
            cv2.putText(
                debug_image,
                text,
                (center[0] - 50, center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # 3D koordinatlar
            pos_text = (
                f"({pose.position.x:.2f}, {pose.position.y:.2f}, {pose.position.z:.2f})"
            )
            cv2.putText(
                debug_image,
                pos_text,
                (center[0] - 80, center[1] + radius + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

        return debug_image

    def image_callback(self, msg):
        if self.camera_matrix is None:
            rospy.logwarn_throttle(5, "Camera calibration not received yet")
            return

        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Kırmızı nesneleri tespit et
            contours, mask = self.detect_red_objects(cv_image)

            # Pose array oluştur
            pose_array = PoseArray()
            pose_array.header = Header()
            pose_array.header.stamp = msg.header.stamp
            pose_array.header.frame_id = (
                msg.header.frame_id if msg.header.frame_id else "camera_link"
            )

            poses_info = []  # Debug için

            for contour in contours:
                result = self.estimate_pipe_pose(contour, cv_image.shape)
                if result is not None:
                    pose, center, radius, distance = result
                    pose_array.poses.append(pose)
                    poses_info.append((pose, center, radius, distance))

            # Pose'ları yayınla
            if len(pose_array.poses) > 0:
                self.pose_pub.publish(pose_array)
                rospy.logdebug(f"Published {len(pose_array.poses)} pipe poses")

            # Debug görüntüsü yayınla
            if self.publish_debug and poses_info:
                debug_image = self.draw_debug_info(cv_image, poses_info)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                debug_msg.header = msg.header
                self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"Error in image processing: {str(e)}")

    def run(self):
        rospy.loginfo("Red Pipe Pose Detector running...")
        rospy.spin()


if __name__ == "__main__":
    try:
        detector = RedPipePoseDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Red Pipe Pose Detector shutting down")
    except Exception as e:
        rospy.logerr(f"Fatal error: {str(e)}")

#!/usr/bin/env python3
# -- coding: utf-8 --

"""
Valve Detector Node
-------------------
Simülasyon ön kamerasından gelen görüntüyü işleyerek:
  1. HSV maskeleme ile turuncu vanayı tespit eder
  2. cv2.fitEllipse ile elips fit eder
  3. Elipsin eksenlerinden eğim açısı (θ) hesaplar
  4. Bilinen gerçek çaptan mesafe hesaplar
  5. valve_stand_link TF frame'ini yayınlar (pozisyon + oryantasyon)
  6. Sap açısını PropsYaw mesajı olarak yayınlar

Topic'ler:
  Subscribes: /taluy/cameras/cam_front/image_underwater
  Publishes:  /valve_detector/debug_image (sensor_msgs/Image)
              /valve_detector/valve_mask (sensor_msgs/Image)
              object_transform_updates (geometry_msgs/TransformStamped)
              props_yaw (auv_msgs/PropsYaw)
"""

import rospy
import math
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from auv_msgs.msg import PropsYaw
from cv_bridge import CvBridge
import tf2_ros
import tf.transformations as tft
import auv_common_lib.vision.camera_calibrations as camera_calibrations


class CameraCalibration:
    """Kamera kalibrasyon verisi ile açı ve mesafe hesaplama."""

    def __init__(self, namespace: str):
        self.calibration = camera_calibrations.CameraCalibrationFetcher(
            namespace, True
        ).get_camera_info()

    def calculate_angles(self, pixel_coordinates: tuple) -> tuple:
        fx = self.calibration.K[0]
        fy = self.calibration.K[4]
        cx = self.calibration.K[2]
        cy = self.calibration.K[5]
        norm_x = (pixel_coordinates[0] - cx) / fx
        norm_y = (pixel_coordinates[1] - cy) / fy
        angle_x = math.atan(norm_x)
        angle_y = math.atan(norm_y)
        return angle_x, angle_y

    def distance_from_width(self, real_width: float, measured_width: float) -> float:
        focal_length = self.calibration.K[0]
        distance = (real_width * focal_length) / measured_width
        return distance


class ValveDetectorNode:
    """
    Ana valve detector node.
    HSV maskeleme -> Elips fit -> Mesafe + Oryantasyon -> TF publish
    """

    def __init__(self):
        rospy.init_node("valve_detector_node", anonymous=True)
        rospy.loginfo("Valve detector node started")

        # ---- Parametreleri yükle ----
        self.hsv_lower = np.array(rospy.get_param("~hsv_lower", [10, 150, 150]))
        self.hsv_upper = np.array(rospy.get_param("~hsv_upper", [25, 255, 255]))
        self.valve_diameter = rospy.get_param("~valve_diameter", 0.240)
        self.handle_width = rospy.get_param("~handle_width", 0.025)
        self.blur_kernel_size = rospy.get_param("~blur_kernel_size", 5)
        self.morph_kernel_size = rospy.get_param("~morph_kernel_size", 5)
        self.min_contour_area = rospy.get_param("~min_contour_area", 100)
        self.min_ellipse_points = rospy.get_param("~min_ellipse_points", 5)
        self.min_aspect_ratio = rospy.get_param("~min_aspect_ratio", 0.2)
        self.max_aspect_ratio = rospy.get_param("~max_aspect_ratio", 2.0)
        self.publish_debug_image = rospy.get_param("~publish_debug_image", True)

        # ---- Kamera kalibrasyonu ----
        self.calibration = CameraCalibration("taluy/cameras/cam_front")

        # ---- OpenCV bridge ----
        self.bridge = CvBridge()

        # ---- TF broadcaster ----
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # ---- Publishers ----
        self.transform_pub = rospy.Publisher(
            "object_transform_updates", TransformStamped, queue_size=10
        )
        self.props_yaw_pub = rospy.Publisher(
            "props_yaw", PropsYaw, queue_size=10
        )
        self.debug_image_pub = rospy.Publisher(
            "valve_detector/debug_image", Image, queue_size=1
        )
        self.mask_pub = rospy.Publisher(
            "valve_detector/valve_mask", Image, queue_size=1
        )

        # ---- Subscriber ----
        # Launch dosyasından topic ismini al (varsayılan: rectified color)
        input_topic = rospy.get_param("~input_topic", "/taluy/cameras/cam_front/image_rect_color")
        
        rospy.Subscriber(
            input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )

        rospy.loginfo(f"Valve detector: Listening on {input_topic}")

    # =====================================================================
    #  ANA CALLBACK
    # =====================================================================
    def image_callback(self, msg):
        """Her frame için çağrılır. Pipeline'ın giriş noktası."""
        # 1) ROS Image -> OpenCV
        rospy.loginfo("Callback tetiklendi, goruntu isleniyor...")
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 2) HSV maskeleme -> turuncu bölgeleri bul
        mask = self.create_orange_mask(frame)

        # 3) En büyük konturu bul
        contour = self.find_largest_contour(mask)

        if contour is None:
            if self.publish_debug_image:
                self.publish_debug(frame, None, 0, 0, None)
            return

        # 4) Ellipse fit
        ellipse = self.fit_ellipse(contour)

        if ellipse is None:
            if self.publish_debug_image:
                self.publish_debug(frame, None, 0, 0, None)
            return

        center, axes, angle = ellipse
        # axes = (büyük eksen uzunluğu, küçük eksen uzunluğu) piksel cinsinden

        # 5) Eğim açısı hesapla: θ = arccos(b / a)
        tilt_angle = self.calculate_tilt_angle(axes)

        # 6) Mesafe hesapla (bilinen çaptan)
        distance = self.estimate_distance(axes)

        if distance is None or distance <= 0:
            return

        # 7) Kameraya göre açıları hesapla (yaw, pitch)
        angle_x, angle_y = self.calibration.calculate_angles(center)

        # 8) 3D pozisyon hesapla (kamera frame'inde)
        x = distance  # ileri (kameradan uzaklık)
        y = -distance * math.tan(angle_x)  # sola/sağa
        z = -distance * math.tan(angle_y)  # yukarı/aşağı

        # 9) Oryantasyon hesapla (elips açısı + eğim -> quaternion)
        orientation_quat = self.calculate_orientation(angle, tilt_angle)

        # 10) TF yayınla
        self.publish_valve_tf(msg.header, x, y, z, orientation_quat)

        # 11) Sap açısını yayınla (PropsYaw)
        handle_angle = self.detect_handle_angle(frame, mask, ellipse)
        if handle_angle is not None:
            self.publish_handle_yaw(handle_angle)

        # 12) Debug görüntüsü yayınla
        if self.publish_debug_image:
            self.publish_debug(frame, ellipse, distance, tilt_angle, handle_angle)

    # =====================================================================
    #  ADIM 2: HSV MASKELEME
    # =====================================================================
    def create_orange_mask(self, frame):
        """BGR frame -> HSV -> turuncu maske."""
        # Gaussian blur ile gürültüyü azalt
        k = self.blur_kernel_size
        blurred = cv2.GaussianBlur(frame, (k, k), 0)

        # BGR -> HSV dönüşümü
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Turuncu renk maskesi
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        # Morfolojik işlemler (küçük delikleri kapat, gürültüyü temizle)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Mask topic yayınla
        if self.mask_pub.get_num_connections() > 0:
            self.mask_pub.publish(
                self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            )

        return mask

    # =====================================================================
    #  ADIM 3: EN BÜYÜK KONTUR
    # =====================================================================
    def find_largest_contour(self, mask):
        """Maskeden en büyük konturu bul."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Alan filtreleme + en büyüğünü seç
        valid_contours = [
            c for c in contours if cv2.contourArea(c) >= self.min_contour_area
        ]

        if not valid_contours:
            return None

        return max(valid_contours, key=cv2.contourArea)

    # =====================================================================
    #  ADIM 4: ELİPS FIT
    # =====================================================================
    def fit_ellipse(self, contour):
        """
        Kontura elips fit et.
        Returns: (center, axes, angle) veya None
            center = (cx, cy) piksel
            axes   = (a, b) piksel - a >= b garanti
            angle  = elipsin büyük ekseninin açısı (derece)
        """
        if len(contour) < self.min_ellipse_points:
            return None

        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (w, h), angle = ellipse

        # OpenCV'de w ve h her zaman sıralı değil, büyük ekseni a olarak al
        a = max(w, h) / 2.0
        b = min(w, h) / 2.0

        if a <= 0:
            return None

        # Aspect ratio kontrolü (çok garip şekilleri reddet)
        aspect_ratio = b / a
        if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
            return None

        return (cx, cy), (a, b), angle

    # =====================================================================
    #  ADIM 5: EĞİM AÇISI
    # =====================================================================
    def calculate_tilt_angle(self, axes):
        """
        Elipsin eksenlerinden vana yüzeyinin eğim açısını hesapla.
        θ = arccos(b / a)
        a = b ise vana tam karşıda (θ = 0)
        a >> b ise vana çok eğik
        """
        a, b = axes
        if a <= 0:
            return 0.0
        ratio = min(b / a, 1.0)  # sayısal hatalardan korun
        return math.acos(ratio)

    # =====================================================================
    #  ADIM 6: MESAFE
    # =====================================================================
    def estimate_distance(self, axes):
        """
        Bilinen vana çapı ve piksel büyüklüğünden mesafe hesapla.
        Büyük ekseni (a) kullan - eğimden bağımsız en doğru ölçü.
        """
        a, b = axes
        # Büyük eksen (a) -> gerçek çapa en yakın (eğimden etkilenmez)
        measured_diameter_px = a * 2.0

        if measured_diameter_px <= 0:
            return None

        return self.calibration.distance_from_width(
            self.valve_diameter, measured_diameter_px
        )

    # =====================================================================
    #  ADIM 9: ORYANTASYON -> QUATERNION
    # =====================================================================
    def calculate_orientation(self, ellipse_angle_deg, tilt_angle_rad):
        """
        Elips açısı ve eğimden vananın 3D oryantasyonunu quaternion olarak hesapla.

        ellipse_angle_deg: Elipsin büyük ekseninin görüntüdeki açısı (derece)
                           -> vananın hangi yöne eğik olduğunu belirler
        tilt_angle_rad:    Eğim açısı (radian), arccos(b/a)
                           -> vananın ne kadar eğik olduğunu belirler
        """
        # Elips açısını radyana çevir
        ellipse_angle_rad = math.radians(ellipse_angle_deg)

        # Vana yüzeyi kameraya bakıyorsa: yaw = 0, pitch = 0
        # Eğim varsa: tilt_angle kadar pitch/yaw, ellipse_angle yönünde
        # Basitleştirilmiş model: eğim pitch olarak uygulanır
        yaw = 0.0
        pitch = tilt_angle_rad
        roll = ellipse_angle_rad

        # Euler -> Quaternion (RPY sırası)
        quat = tft.quaternion_from_euler(roll, pitch, yaw)

        return Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

    # =====================================================================
    #  ADIM 10: TF YAYINLA
    # =====================================================================
    def publish_valve_tf(self, header, x, y, z, orientation):
        """valve_stand_link TF frame'ini yayınla."""
        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = header.frame_id
        t.child_frame_id = "valve_stand_link"
        t.transform.translation = Vector3(x=x, y=y, z=z)
        t.transform.rotation = orientation

        # TF broadcast
        self.tf_broadcaster.sendTransform(t)

        # Topic'e de yayınla (object_map_tf_server için)
        self.transform_pub.publish(t)

    # =====================================================================
    #  ADIM 11: SAP AÇISI TESPİTİ
    # =====================================================================
    def detect_handle_angle(self, frame, mask, ellipse):
        """
        Vana sapının açısını tespit et.
        Elips içindeki çizgisel yapıyı bularak sapın yönünü hesaplar.

        Returns: sap açısı (derece) veya None
        """
        center, axes, angle = ellipse
        a, b = axes
        cx, cy = center

        # Elips bölgesini maskele (sadece elips içini al)
        ellipse_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.ellipse(
            ellipse_mask,
            (int(cx), int(cy)),
            (int(a), int(b)),
            angle, 0, 360, 255, -1
        )

        # Elips içindeki orijinal frame'den gri tonlama
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.bitwise_and(gray, gray, mask=ellipse_mask)

        # Kenar tespiti ile sapı bul
        edges = cv2.Canny(roi, 50, 150)

        # Hough çizgi tespiti ile sapın açısını bul
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=30,
            minLineLength=int(a * 0.4),
            maxLineGap=10
        )

        if lines is None or len(lines) == 0:
            return None

        # En uzun çizgiyi sap olarak kabul et
        best_line = None
        best_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > best_length:
                best_length = length
                best_line = line[0]

        if best_line is None:
            return None

        x1, y1, x2, y2 = best_line
        handle_angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        return handle_angle

    # =====================================================================
    #  ADIM 11b: PROPSYAW YAYINLA
    # =====================================================================
    def publish_handle_yaw(self, handle_angle_deg):
        """Sap açısını PropsYaw mesajı olarak yayınla."""
        msg = PropsYaw()
        msg.angle = handle_angle_deg
        msg.object = "valve_handle"
        self.props_yaw_pub.publish(msg)

    # =====================================================================
    #  ADIM 12: DEBUG GÖRÜNTÜSÜ
    # =====================================================================
    def publish_debug(self, frame, ellipse, distance, tilt_angle, handle_angle):
        """Debug amaçlı annotated görüntü yayınla."""
        if self.debug_image_pub.get_num_connections() == 0:
            return

        debug_frame = frame.copy()

        # Eğer elips/vana bulunamadıysa: Sadece "VANA YOK" yaz
        if ellipse is None:
            cv2.putText(
                debug_frame, "VANA BULUNAMADI (RENK HATASI?)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
            )
            return

        center, axes, angle = ellipse
        a, b = axes

        # Elipsi çiz
        cv2.ellipse(
            debug_frame,
            (int(center[0]), int(center[1])),
            (int(a), int(b)),
            angle, 0, 360, (0, 255, 0), 2
        )

        # Merkez noktası
        cv2.circle(debug_frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)

        # Bilgileri yaz
        info_lines = [
            f"Dist: {distance:.2f}m",
            f"Tilt: {math.degrees(tilt_angle):.1f} deg",
            f"a/b: {a:.0f}/{b:.0f} px",
        ]
        if handle_angle is not None:
            info_lines.append(f"Handle: {handle_angle:.1f} deg")

        y_offset = 30
        for line in info_lines:
            cv2.putText(
                debug_frame, line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25

        self.debug_image_pub.publish(
            self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
        )


def main():
    ValveDetectorNode()
    rospy.spin()


if __name__ == "__main__":
    main()

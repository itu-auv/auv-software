#!/usr/bin/env python3
"""
ArUco Auto-Detect & Pose Estimation Node
Bu kod otomatik olarak dogru ArUco sozlugunu bulur ve pose hesabi yapar.
OpenCV 4.2 (Jetson/Ubuntu 18/20) uyumludur.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
import tf.transformations

class ArucoAutoEstimator:
    def __init__(self):
        rospy.init_node('aruco_auto_estimator', anonymous=True)
        
        # --- params ---
        self.marker_size = rospy.get_param('~marker_size', 0.1)
        self.camera_frame = "usb_cam" 
        
        # OpenCV 3.x ve 4.x (pre-4.7) 
        self.ARUCO_DICTS = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
        }
        
        self.current_dict_name = None
        self.current_dict_obj = None
        
        # --- ROS SETUP ---
        self.bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.pose_pub = rospy.Publisher('~detected_pose', PoseStamped, queue_size=10)
        self.debug_pub = rospy.Publisher('~debug_image', Image, queue_size=1)
        #camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.cam_info_sub = rospy.Subscriber('/taluy/cameras/cam_front/camera_info', CameraInfo, self.cam_info_cb)
        self.img_sub = rospy.Subscriber('/taluy/cameras/cam_front/image_raw', Image, self.image_cb)
        
        rospy.loginfo("ArUco Auto-Detector baslatildi. Marker araniyor...")

    def cam_info_cb(self, msg):
        self.camera_matrix = np.array(msg.K).reshape(3, 3)
        self.dist_coeffs = np.array(msg.D)
        self.camera_frame = msg.header.frame_id

    def get_dummy_calibration(self, height, width):
        """Eger camera_info gelmezse, kodu calistirmak icin sahte kalibrasyon uretir"""
        focal_length = width  
        center_x = width / 2
        center_y = height / 2
        cam_mat = np.array([
            [focal_length, 0, center_x],
            [0, focal_length, center_y],
            [0, 0, 1]
        ], dtype=np.float32)
        dist = np.zeros((5, 1))
        return cam_mat, dist

    def image_cb(self, msg):
        rospy.loginfo_once("Image callback")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatasi: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        if self.camera_matrix is None:
            h, w = gray.shape
            cam_mat, dist_coef = self.get_dummy_calibration(h, w)
            rospy.logwarn_once("UYARI: Camera Info gelmedi! Dummy kalibrasyon kullaniliyor.")
        else:
            cam_mat, dist_coef = self.camera_matrix, self.dist_coeffs

        # --- ARUCO TARAMA DONGUSU ---
        # Eger daha once bir sozluk bulduysak once onu deneriz, yoksa hepsini deneriz
        found_ids = None
        found_corners = None
        
        # Denenecek sozlukler listesi (Once bildigimiz, sonra hepsi)
        dictionaries_to_try = []
        if self.current_dict_obj is not None:
            dictionaries_to_try.append((self.current_dict_name, self.current_dict_obj))
        else:
            # Hepsini dene
            for name, enum_val in self.ARUCO_DICTS.items():
                dictionaries_to_try.append((name, cv2.aruco.Dictionary_get(enum_val)))

        detector_params = cv2.aruco.DetectorParameters_create()
        # Parametreleri biraz gevsetelim ki kolay bulsun
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        for name, dict_obj in dictionaries_to_try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, dict_obj, parameters=detector_params
            )
            
            if ids is not None and len(ids) > 0:
                found_ids = ids
                found_corners = corners
                
                # Eger yeni bir sozluk kesfettiysek kaydedelim
                if self.current_dict_name != name:
                    self.current_dict_name = name
                    self.current_dict_obj = dict_obj
                    rospy.loginfo(f"----------------------------------------")
                    rospy.loginfo(f"BASARI! Marker bulundu. Sozluk: {name}")
                    rospy.loginfo(f"----------------------------------------")
                break
        
        # --- GORSELLESTIRME VE YAYIN ---
        debug_img = cv_image.copy()
        
        if found_ids is not None:
            # Marker ciz
            cv2.aruco.drawDetectedMarkers(debug_img, found_corners, found_ids)
            
            # Pose Estimate
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                found_corners, self.marker_size, cam_mat, dist_coef
            )
            
            for i in range(len(found_ids)):
                # Eksen ciz
                cv2.drawFrameAxes(
                    debug_img, cam_mat, dist_coef, 
                    rvecs[i], tvecs[i], self.marker_size * 0.5
                )
                
                # TF ve Pose yayinla
                self.publish_tf_pose(found_ids[i][0], rvecs[i][0], tvecs[i][0], msg.header)
                
                # Ekrana mesafe yaz
                dist = np.linalg.norm(tvecs[i][0])
                cv2.putText(debug_img, f"ID:{found_ids[i][0]} Dist:{dist:.2f}m", 
                           (10, 70 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Ekrana hangi sozlugu kullandigimizi yazalim
            cv2.putText(debug_img, f"DICT: {self.current_dict_name}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # Bulunamadiysa
            self.current_dict_name = None # Kaydi sil ki tekrar tarasin
            cv2.putText(debug_img, "ARANIYOR...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Debug goruntusunu yayinla
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

    def publish_tf_pose(self, id, rvec, tvec, header):
        # Rotasyon Vektoru -> Matris -> Quaternion
        rot_mat, _ = cv2.Rodrigues(rvec)
        
        # Homojen matris olustur
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rot_mat
        transform_matrix[0:3, 3] = tvec
        
        # Quaternion hesabi
        quat = tf.transformations.quaternion_from_matrix(transform_matrix)
        
        # TF Mesaji
        t = TransformStamped()
        t.header.stamp = header.stamp
        # Eger camera_frame bos veya hataliysa manuel 'usb_cam' yapalim
        t.header.frame_id = self.camera_frame if self.camera_frame else "usb_cam"
        t.child_frame_id = f"aruco_{id}"
        
        t.transform.translation.x = tvec[0]
        t.transform.translation.y = tvec[1]
        t.transform.translation.z = tvec[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ArucoAutoEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
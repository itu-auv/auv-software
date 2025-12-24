#!/usr/bin/env python3
"""
Webcam Debug Publisher
Publishes laptop webcam images to ROS for testing aruco_pose_estimator.py locally.
"""

import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


class WebcamPublisher:
    def __init__(self):
        rospy.init_node('webcam_debug_publisher', anonymous=True)
        
        # Parameters
        self.camera_ns = rospy.get_param('~camera_ns', 'taluy/cameras/cam_front')
        self.device_id = rospy.get_param('~device_id', 0)  # /dev/video0
        self.fps = rospy.get_param('~fps', 30)
        self.frame_id = rospy.get_param('~frame_id', 'taluy/base_link/front_camera_optical_link')
        
        # OpenCV capture
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            rospy.logerr(f"Cannot open webcam device {self.device_id}")
            raise RuntimeError("Webcam not available")
        
        # Get actual camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rospy.loginfo(f"Webcam opened: {self.width}x{self.height}")
        
        self.bridge = CvBridge()
        
        # Publishers - match the topics aruco_pose_estimator.py expects
        # DEBUG: Using webcam-specific topics instead of camera_ns
        # image_topic = f'/{self.camera_ns}/image_raw'
        # info_topic = f'/{self.camera_ns}/camera_info'
        image_topic = '/webcam/image_raw'
        info_topic = '/webcam/camera_info'
        
        self.image_pub = rospy.Publisher(image_topic, Image, queue_size=1)
        self.info_pub = rospy.Publisher(info_topic, CameraInfo, queue_size=1)
        
        rospy.loginfo(f"Publishing to: {image_topic}")
        rospy.loginfo(f"Publishing to: {info_topic}")
        
        # Create a basic camera info (approximate webcam calibration)
        self.camera_info = self._create_default_camera_info()
        
        # Timer for publishing
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.fps), self.publish_frame)
        
        rospy.loginfo("Webcam debug publisher started!")

    def _create_default_camera_info(self):
        """Create approximate camera info for a typical webcam."""
        info = CameraInfo()
        info.header.frame_id = self.frame_id
        info.width = self.width
        info.height = self.height
        
        # Approximate intrinsics for a typical webcam (adjust if needed)
        # Focal length ~ width (common approximation for ~60 degree FOV)
        fx = fy = float(self.width)
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        info.K = [fx, 0, cx,
                  0, fy, cy,
                  0, 0, 1]
        
        info.D = [0.0, 0.0, 0.0, 0.0, 0.0]  # No distortion
        
        info.R = [1, 0, 0,
                  0, 1, 0,
                  0, 0, 1]
        
        info.P = [fx, 0, cx, 0,
                  0, fy, cy, 0,
                  0, 0, 1, 0]
        
        info.distortion_model = "plumb_bob"
        
        return info

    def publish_frame(self, event):
        ret, frame = self.cap.read()
        if not ret:
            rospy.logwarn_throttle(5, "Failed to read from webcam")
            return
        
        now = rospy.Time.now()
        
        # Publish image
        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        img_msg.header.stamp = now
        img_msg.header.frame_id = self.frame_id
        self.image_pub.publish(img_msg)
        
        # Publish camera info
        self.camera_info.header.stamp = now
        self.info_pub.publish(self.camera_info)

    def shutdown(self):
        self.cap.release()
        rospy.loginfo("Webcam released")

    def run(self):
        rospy.on_shutdown(self.shutdown)
        rospy.spin()


if __name__ == '__main__':
    try:
        node = WebcamPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass

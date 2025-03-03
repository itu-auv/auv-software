#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion

class DynamicRollCompensationNode:
    def __init__(self):
        rospy.init_node('dynamic_roll_compensation', anonymous=True)
        
        self.bridge = CvBridge()
        self.roll_angle_deg = 0.0
        
        # Subscribe to the IMU data
        self.imu_subscriber = rospy.Subscriber(
            "imu/data", Imu, self.imu_callback, tcp_nodelay=True
        )
        
        # Subscribe to the raw camera image topic
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.image_callback)
        
        # Publisher for the corrected (or unmodified) image
        self.image_pub = rospy.Publisher("camera/image_corrected", Image, queue_size=10)
        
        rospy.loginfo("Dynamic roll compensation node started.")

    def imu_callback(self, imu_msg):
        """
        Callback to update the roll angle from the IMU data.
        """
        # Extract the orientation quaternion from the IMU message
        q = [
            imu_msg.orientation.x,
            imu_msg.orientation.y,
            imu_msg.orientation.z,
            imu_msg.orientation.w
        ]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion(q)
        
        # Convert roll from radians to degrees (OpenCV uses degrees for rotation)
        self.roll_angle_deg = math.degrees(roll)
        rospy.logdebug("Updated IMU roll angle: %.2f degrees", self.roll_angle_deg)

    def image_callback(self, img_msg):
        """
        Callback to process the incoming camera image using the dynamic roll angle.
        If the absolute roll angle is greater than 30 degrees, the image is published without correction.
        """
        try:
            # Convert the ROS image message to an OpenCV image (assumed "bgr8")
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        # Check if the roll angle exceeds 30 degrees
        if abs(self.roll_angle_deg) > 30:
            rospy.logwarn("Roll angle (%.2f) exceeds 30 degrees. Publishing unmodified image.", self.roll_angle_deg)
            # Publish the original image message without modification
            self.image_pub.publish(img_msg)
            return
        
        # Get image dimensions and compute the center of the image
        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)
        
        # Compute the rotation matrix for the current roll compensation angle.
        # A positive angle rotates the image counter-clockwise.
        M = cv2.getRotationMatrix2D(center, -self.roll_angle_deg, 1.0)
        
        # Rotate the image using the computed matrix
        rotated_image = cv2.warpAffine(cv_image, M, (w, h))
        
        try:
            # Convert the rotated OpenCV image back to a ROS image message
            corrected_img_msg = self.bridge.cv2_to_imgmsg(rotated_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        # Retain the original header
        corrected_img_msg.header = img_msg.header

        # Publish the roll-compensated image
        self.image_pub.publish(corrected_img_msg)
        rospy.logdebug("Published corrected image with roll compensation (%.2f degrees).", self.roll_angle_deg)

def main():
    try:
        node = DynamicRollCompensationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dynamic roll compensation node terminated.")

if __name__ == '__main__':
    main()

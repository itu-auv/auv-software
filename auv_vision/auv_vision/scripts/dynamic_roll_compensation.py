#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import math
import angles  # ROS angles library
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_quaternion

class DynamicRollCompensationNode:
    def __init__(self):
        
        rospy.init_node('dynamic_roll_compensation', anonymous=True)
        rospy.loginfo("Dynamic roll compensation node started.")
        
        self.bridge = CvBridge()
        self.roll_angle_deg = 0.0

        self.odom_subscriber = rospy.Subscriber("odometry", Odometry, self.odom_callback, tcp_nodelay=True)
        
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("camera/image_corrected", Image, queue_size=10)

    def odom_callback(self, odom_msg):

        q = [
            odom_msg.pose.pose.orientation.x,
            odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z,
            odom_msg.pose.pose.orientation.w
        ]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion(q)
        
        # Compute correction angle (in radians) as the shortest angular distance from roll to 0.
        correction_rad = angles.shortest_angular_distance(roll, 0.0)
        self.roll_angle_deg = math.degrees(correction_rad)
        rospy.logdebug("Computed correction angle (radians): %.2f", correction_rad)

    def image_callback(self, img_msg):

        try:
            # Convert the ROS Image message to an OpenCV image (assumed "bgr8" encoding)
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
            return

        # Get image dimensions and compute the center
        (h, w) = cv_image.shape[:2]
        center = (w // 2, h // 2)
        rospy.logdebug("Image dimensions: width=%d, height=%d, center=%s", w, h, center)
        
        # Compute the rotation matrix using the computed correction angle
        M = cv2.getRotationMatrix2D(center, self.roll_angle_deg, 1.0)
        rospy.logdebug("Computed rotation matrix: %s", M)
        
        # Rotate the image using the computed matrix
        rotated_image = cv2.warpAffine(cv_image, M, (w, h))
        rospy.logdebug("Applied cv2.warpAffine to rotate the image.")

        try:
            # Convert the rotated OpenCV image back to a ROS Image message
            corrected_img_msg = self.bridge.cv2_to_imgmsg(rotated_image, "bgr8")
            rospy.logdebug("Converted rotated image to ROS Image message.")
        except CvBridgeError as e:
            rospy.logerr("CvBridge error while converting rotated image: %s", e)
            return

        # Retain the original header information
        corrected_img_msg.header = img_msg.header

        # Publish the roll-corrected image
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

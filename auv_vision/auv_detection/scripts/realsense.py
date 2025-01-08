#!/usr/bin/env python3


import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class DepthImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.image_callback)
    
    def image_callback(self, msg):
        # Image mesajını OpenCV formatına dönüştürme
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Görüntüyü göster
        cv2.imshow("Depth Image", cv_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('depth_image_processor')
    depth_processor = DepthImageProcessor()
    rospy.spin()

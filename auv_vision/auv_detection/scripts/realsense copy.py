#!/usr/bin/env python3
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
class DepthTo3D:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.image_callback)
    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
        # Görüntü boyutları
        height, width = cv_image.shape
        
        # Derinlik bilgilerini kullanarak 3D koordinatları hesaplayabiliriz
        for y in range(height):
            for x in range(width):
                depth = cv_image[y, x]
                if depth > 0:  # Geçerli bir derinlik varsa
                    # Kamera iç parametrelerini kullanarak (fx, fy, cx, cy) bu derinliği 3D koordinatlara dönüştürebilirsiniz
                    x3d = (x - 306) * depth / 432 #fx
                    y3d = (y - 221) * depth / 567 #fy
                    z3d = depth
                    
                    # 3D koordinatları işleme
                    print(f"3D Koordinatlar: x: {x3d}, y: {y3d}, z: {z3d}")

if __name__ == '__main__':
    rospy.init_node('depth_to_3d')
    depth_processor = DepthTo3D()
    rospy.spin()
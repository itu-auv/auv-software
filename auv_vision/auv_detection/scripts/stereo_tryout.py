#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class StereoImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()
        rospy.loginfo("Starting Stereo Image Processor...")
        
        # Subscribe to both infrared camera topics
        self.left_sub = rospy.Subscriber('/camera/infra1/image_rect_raw', 
                                       Image, 
                                       self.left_callback)
        self.right_sub = rospy.Subscriber('/camera/infra2/image_rect_raw', 
                                        Image, 
                                        self.right_callback)
        
        rospy.loginfo("Subscribed to infrared topics")
        self.left_image = None
        self.right_image = None

    def left_callback(self, msg):
        try:
            rospy.loginfo_throttle(1, "Receiving left camera images")
            self.left_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.show_images()
        except CvBridgeError as e:
            rospy.logerr(e)

    def right_callback(self, msg):
        try:
            rospy.loginfo_throttle(1, "Receiving right camera images")
            self.right_image = self.bridge.imgmsg_to_cv2(msg, "mono8")
            self.show_images()
        except CvBridgeError as e:
            rospy.logerr(e)

    def show_images(self):
        if self.left_image is not None and self.right_image is not None:
            # Display the images side by side
            stereo_image = cv2.hconcat([self.left_image, self.right_image])
            cv2.imshow('Stereo Cameras (Left | Right)', stereo_image)
            cv2.waitKey(1)

def main():
    rospy.init_node('stereo_image_processor')
    processor = StereoImageProcessor()
    rospy.loginfo("Stereo Image Processor is running. Press Ctrl+C to exit.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

def main():
    rospy.init_node('realsense_ir_y16_publisher', anonymous=True)

    # Publishers for left and right infrared images (raw Y16)
    pub_left = rospy.Publisher("/camera/infra1/image_raw", Image, queue_size=10)
    pub_right = rospy.Publisher("/camera/infra2/image_raw", Image, queue_size=10)

    bridge = CvBridge()

    # Configure RealSense pipeline and streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable left infrared (index 1) as Y16 stream:
    config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y16, 30)
    # Enable right infrared (index 2) as Y16 stream:
    config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y16, 30)

    # Start the pipeline
    profile = pipeline.start(config)
    rospy.loginfo("RealSense infrared streams started (Y16).")

    rate = rospy.Rate(30)  # 30 Hz

    try:
        while not rospy.is_shutdown():
            # Wait for a coherent pair of frames: left and right infrared frames
            frames = pipeline.wait_for_frames()
            left_frame = frames.get_infrared_frame(1)   # Left IR
            right_frame = frames.get_infrared_frame(2)  # Right IR

            if not left_frame or not right_frame:
                continue

            # Convert frame data to numpy arrays (each will be 16-bit, shape: [height, width])
            left_image = np.asanyarray(left_frame.get_data())
            right_image = np.asanyarray(right_frame.get_data())

            # Convert numpy arrays to ROS Image messages with encoding "mono16"
            left_msg = bridge.cv2_to_imgmsg(left_image, encoding="mono16")
            right_msg = bridge.cv2_to_imgmsg(right_image, encoding="mono16")

            # Publish the images
            pub_left.publish(left_msg)
            pub_right.publish(right_msg)

            rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        pipeline.stop()
        rospy.loginfo("RealSense pipeline stopped.")

if __name__ == '__main__':
    main()

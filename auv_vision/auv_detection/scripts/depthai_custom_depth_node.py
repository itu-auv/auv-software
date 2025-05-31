#!/usr/bin/env python3
import rospy
import depthai as dai
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2


class DepthAIWrapper:
    def __init__(self):
        rospy.init_node("depthai_node", anonymous=True)
        self.pub = rospy.Publisher("/oak/depth_map", Image, queue_size=1)
        self.bridge = CvBridge()

        self.pipeline = dai.Pipeline()

        # Color camera
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(640, 480)  # model expects 640x480
        cam.setInterleaved(False)
        cam.setFps(15)

        # Neural network
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(rospy.get_param("~model_path"))

        cam.preview.link(nn.input)

        # XLinkOut
        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("nn")
        nn.out.link(xout.input)

        # Connect to device and start
        self.device = dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.queue.has():
                data = self.queue.get()
                # Layer name discovered to be '707'
                output = np.array(data.getLayerFp16("707"), dtype=np.float32)
                output = output.reshape((240, 320))

                # Normalize for visualization (optional)
                normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
                image_msg = self.bridge.cv2_to_imgmsg(
                    normalized.astype(np.uint8), encoding="mono8"
                )

                header = Header()
                header.stamp = rospy.Time.now()
                header.frame_id = "oak_rgb_camera_frame"
                image_msg.header = header

                self.pub.publish(image_msg)
            rate.sleep()


if __name__ == "__main__":
    try:
        wrapper = DepthAIWrapper()
        wrapper.run()
    except rospy.ROSInterruptException:
        pass

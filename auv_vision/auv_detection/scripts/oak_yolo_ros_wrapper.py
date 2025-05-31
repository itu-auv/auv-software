#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import depthai as dai
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from cv_bridge import CvBridge
import json
import os


class OAKYoloROSWrapper:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("oak_yolo_ros_wrapper", anonymous=True)

        # Load parameters from ROS parameter server
        self.model_path = rospy.get_param("~model_path")
        self.config_path = rospy.get_param("~config_path")
        self.conf_threshold = rospy.get_param("~conf_threshold", 0.5)
        self.iou_threshold = rospy.get_param("~iou_threshold", 0.4)
        self.camera_resolution = rospy.get_param(
            "~camera_resolution", "1080p"
        )  # Options: '1080p', '4K'

        # Validate model and config paths
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model file {self.model_path} does not exist")
            rospy.signal_shutdown("Invalid model path")
            return
        if not os.path.exists(self.config_path):
            rospy.logerr(f"Config file {self.config_path} does not exist")
            rospy.signal_shutdown("Invalid config path")
            return

        # Load YOLO configuration
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        self.labels = self.config.get("labels", [])
        self.input_size = self.config.get("input_size", [416, 416])  # [width, height]
        rospy.loginfo(
            f"Loaded config: labels={self.labels}, input_size={self.input_size}"
        )

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # ROS Publishers
        self.image_pub = rospy.Publisher("~image_raw", Image, queue_size=10)
        self.detections_pub = rospy.Publisher(
            "~detections", Detection2DArray, queue_size=10
        )
        self.image_with_detections_pub = rospy.Publisher(
            "~image_with_detections", Image, queue_size=10
        )

        # Initialize DepthAI pipeline
        self.pipeline = dai.Pipeline()
        self.setup_pipeline()

        # Start the device
        try:
            self.device = dai.Device(self.pipeline)
        except Exception as e:
            rospy.logerr(f"Failed to initialize OAK device: {str(e)}")
            rospy.signal_shutdown("Device initialization failed")
            return

        self.rgb_queue = self.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )
        self.nn_queue = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        rospy.loginfo("OAK YOLO ROS Wrapper initialized")

    def setup_pipeline(self):
        # Create nodes
        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        nn = self.pipeline.create(dai.node.YoloDetectionNetwork)
        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_nn = self.pipeline.create(dai.node.XLinkOut)

        # Set output stream names
        xout_rgb.setStreamName("rgb")
        xout_nn.setStreamName("nn")

        # Configure camera
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_1080_P
            if self.camera_resolution == "1080p"
            else dai.ColorCameraProperties.SensorResolution.THE_4_K
        )
        cam_rgb.setPreviewSize(self.input_size[0], self.input_size[1])
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # Configure YOLO neural network with .blob model
        nn.setBlobPath(self.model_path)
        nn.setConfidenceThreshold(self.conf_threshold)
        nn.setIouThreshold(self.iou_threshold)
        nn.setNumInferenceThreads(2)
        nn.setCoordinateSize(4)
        nn.setAnchors(self.config.get("anchors", []))
        nn.setAnchorMasks(self.config.get("anchor_masks", {}))
        nn.setNumClasses(len(self.labels))
        nn.input.setBlocking(False)

        # Link nodes
        cam_rgb.preview.link(nn.input)
        cam_rgb.preview.link(xout_rgb.input)
        nn.out.link(xout_nn.input)

    def run(self):
        rate = rospy.Rate(30)  # 30 Hz
        while not rospy.is_shutdown():
            # Get RGB frame
            in_rgb = self.rgb_queue.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                # Convert to ROS Image message
                image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                image_msg.header.stamp = rospy.Time.now()
                self.image_pub.publish(image_msg)
                rospy.logdebug("Published RGB image")

            # Get YOLO detections
            in_nn = self.nn_queue.tryGet()
            if in_nn is not None:
                detections = in_nn.detections
                rospy.loginfo(f"Received {len(detections)} detections from NN queue")
                for i, det in enumerate(detections):
                    rospy.logdebug(
                        f"Detection {i}: class_id={det.label}, confidence={det.confidence}, "
                        f"bbox=[{det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}]"
                    )

                detection_array = Detection2DArray()
                detection_array.header.stamp = rospy.Time.now()

                image_with_boxes = frame.copy()

                for det in detections:
                    detection = Detection2D()
                    bbox = BoundingBox2D()
                    bbox.center.x = (det.xmin + det.xmax) / 2.0
                    bbox.center.y = (det.ymin + det.ymax) / 2.0
                    bbox.size_x = det.xmax - det.xmin
                    bbox.size_y = det.ymax - det.ymin
                    detection.bbox = bbox
                    detection.results.append(
                        self.create_object_hypothesis(det.label, det.confidence)
                    )
                    detection_array.detections.append(detection)

                    x1 = int(det.xmin * frame.shape[1])
                    y1 = int(det.ymin * frame.shape[0])
                    x2 = int(det.xmax * frame.shape[1])
                    y2 = int(det.ymax * frame.shape[0])

                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label_text = (
                        self.labels[det.label]
                        if det.label < len(self.labels)
                        else str(det.label)
                    )
                    cv2.putText(
                        image_with_boxes,
                        f"{label_text} {det.confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                self.detections_pub.publish(detection_array)
                image_with_boxes_msg = self.bridge.cv2_to_imgmsg(
                    image_with_boxes, encoding="bgr8"
                )
                image_with_boxes_msg.header.stamp = rospy.Time.now()
                self.image_with_detections_pub.publish(image_with_boxes_msg)
                rospy.loginfo(f"Published {len(detection_array.detections)} detections")

            else:
                rospy.logdebug("No NN data received from queue")

            rate.sleep()

    def create_object_hypothesis(self, class_id, confidence):
        from vision_msgs.msg import ObjectHypothesisWithPose

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.id = class_id
        hypothesis.score = confidence
        if class_id < len(self.labels):
            hypothesis.id = class_id  # Use label name
        return hypothesis

    def shutdown(self):
        self.device.close()
        rospy.loginfo("OAK YOLO ROS Wrapper shutdown")


if __name__ == "__main__":
    try:
        wrapper = OAKYoloROSWrapper()
        wrapper.run()
    except rospy.ROSInterruptException:
        wrapper.shutdown()

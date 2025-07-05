#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np

from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import Detection2D


class YoloHandler:
    def __init__(self):
        rospy.init_node("front_bottom_yolo_handler_node")

        # Check if CUDA is available
        self.gpu_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        rospy.loginfo(f"GPU acceleration available: {self.gpu_available}")

        # Initialize parameters
        self._init_params()
        self.bridge = CvBridge()

        # Pre-allocate GPU memory if available
        if self.gpu_available:
            self.canvas_gpu = cv2.cuda_GpuMat()
            self.front_gpu = cv2.cuda_GpuMat()
            self.bottom_gpu = cv2.cuda_GpuMat()

        # Setup publishers and subscribers
        self._setup_publishers()
        self._setup_subscribers()

    def _init_params(self):
        """Initialize ROS parameters with defaults"""
        self.canvas_w = rospy.get_param("~canvas_width", 1280)
        self.canvas_h = rospy.get_param("~canvas_height", 480)
        self.split_x = self.canvas_w // 2
        self.sync_slop = rospy.get_param(
            "~sync_slop", 0.1
        )  # Time sync tolerance in seconds
        self.queue_size = rospy.get_param("~queue_size", 10)

    def _setup_publishers(self):
        """Initialize all publishers"""
        self.merged_pub = rospy.Publisher("/merged_image", Image, queue_size=1)
        self.pub_front = rospy.Publisher(
            "/yolo/front_view_detections", YoloResult, queue_size=1
        )
        self.pub_bottom = rospy.Publisher(
            "/yolo/bottom_view_detections", YoloResult, queue_size=1
        )

    def _setup_subscribers(self):
        """Initialize all subscribers and synchronizers"""
        # Image sync & merger
        sub_front = Subscriber("cam_front/image_raw", Image)
        sub_bottom = Subscriber("cam_bottom/image_raw", Image)

        ats = ApproximateTimeSynchronizer(
            [sub_front, sub_bottom], queue_size=self.queue_size, slop=self.sync_slop
        )
        ats.registerCallback(self.image_callback)

        # YOLO results subscriber
        rospy.Subscriber("/yolo_result", YoloResult, self.yolo_callback)

    def image_callback(self, front_msg, bottom_msg):
        """
        Callback for synchronized front and bottom camera images.
        Merges them into a single image and publishes it.
        """
        try:
            # Convert ROS images to OpenCV format
            img_front = self.bridge.imgmsg_to_cv2(front_msg, "bgr8")
            img_bottom = self.bridge.imgmsg_to_cv2(bottom_msg, "bgr8")

            if self.gpu_available:
                # Upload images to GPU
                self.front_gpu.upload(img_front)
                self.bottom_gpu.upload(img_bottom)

                # Create blank canvas on GPU
                self.canvas_gpu = cv2.cuda_GpuMat(
                    self.canvas_h, self.canvas_w, cv2.CV_8UC3
                )
                self.canvas_gpu.setTo(0)

                # Place images on canvas using GPU operations
                self._place_image_on_canvas_gpu(
                    self.canvas_gpu, self.front_gpu, 0, self.split_x
                )
                self._place_image_on_canvas_gpu(
                    self.canvas_gpu, self.bottom_gpu, self.split_x, self.canvas_w
                )

                # Download result from GPU
                canvas = self.canvas_gpu.download()
            else:
                # CPU fallback
                canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
                self._place_image_on_canvas(canvas, img_front, 0, self.split_x)
                self._place_image_on_canvas(
                    canvas, img_bottom, self.split_x, self.canvas_w
                )

            # Convert back to ROS image and publish
            out = self.bridge.cv2_to_imgmsg(canvas, "bgr8")
            out.header = front_msg.header
            self.merged_pub.publish(out)

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge error in image_callback: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in image_callback: {e}")

    def _place_image_on_canvas_gpu(self, canvas_gpu, image_gpu, x_start, x_end):
        """
        GPU version of image placement on canvas
        """
        img_h, img_w = image_gpu.size()
        available_w = x_end - x_start

        # Calculate offsets for centering
        x_offset = x_start + (available_w - img_w) // 2
        y_offset = (self.canvas_h - img_h) // 2

        # Ensure we don't go out of bounds
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # Create ROI on canvas
        canvas_roi = cv2.cuda_GpuMat(
            canvas_gpu, (y_offset, y_offset + img_h), (x_offset, x_offset + img_w)
        )

        # Copy image to ROI
        image_gpu.copyTo(canvas_roi)

    def _place_image_on_canvas(self, canvas, image, x_start, x_end):
        """
        Helper method to place an image on the canvas with proper centering
        """
        img_h, img_w = image.shape[:2]
        available_w = x_end - x_start

        # Calculate offsets for centering
        x_offset = x_start + (available_w - img_w) // 2
        y_offset = (self.canvas_h - img_h) // 2

        # Ensure we don't go out of bounds
        x_offset = max(0, x_offset)
        y_offset = max(0, y_offset)

        # Place the image
        canvas[y_offset : y_offset + img_h, x_offset : x_offset + img_w] = image

    def yolo_callback(self, msg: YoloResult):
        """
        Callback for YOLO detections. Splits detections into front and bottom views.
        """
        try:
            front_result = YoloResult()
            bottom_result = YoloResult()

            # Copy headers and masks
            front_result.header = msg.header
            bottom_result.header = msg.header
            front_result.masks = msg.masks
            bottom_result.masks = msg.masks

            # Copy detection headers
            front_result.detections.header = msg.detections.header
            bottom_result.detections.header = msg.detections.header

            # Split detections based on x-coordinate
            for det in msg.detections.detections:
                if det.bbox.center.x < self.split_x:
                    front_result.detections.detections.append(
                        self._shift_detection(det, shift_x=0)
                    )
                else:
                    bottom_result.detections.detections.append(
                        self._shift_detection(det, shift_x=-self.split_x)
                    )

            # Publish results
            self.pub_front.publish(front_result)
            self.pub_bottom.publish(bottom_result)

        except Exception as e:
            rospy.logerr(f"Error in yolo_callback: {e}")

    def _shift_detection(self, det: Detection2D, shift_x=0):
        """
        Helper method to create a new detection with shifted coordinates
        """
        new_det = Detection2D()
        new_det.header = det.header
        new_det.results = det.results
        new_det.source_img = det.source_img

        new_det.bbox.center.x = det.bbox.center.x + shift_x
        new_det.bbox.center.y = det.bbox.center.y
        new_det.bbox.size_x = det.bbox.size_x
        new_det.bbox.size_y = det.bbox.size_y

        return new_det

    def spin(self):
        """Main loop"""
        rospy.spin()


if __name__ == "__main__":
    try:
        handler = YoloHandler()
        handler.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")

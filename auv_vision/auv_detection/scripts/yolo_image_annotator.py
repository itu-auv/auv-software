#!/usr/bin/env python3

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from std_srvs.srv import SetBool, SetBoolResponse
from ultralytics_ros.msg import YoloResult
from cv_bridge import CvBridge
import message_filters


class YoloImageAnnotator:
    def __init__(self):
        rospy.init_node("yolo_image_annotator", anonymous=True)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Get parameters
        self.save_directory = rospy.get_param(
            "~save_directory", "/home/agxorin/yolo_annotated_images"
        )
        self.font_scale = rospy.get_param("~font_scale", 0.7)
        self.font_thickness = rospy.get_param("~font_thickness", 2)

        self.class_names = rospy.get_param(
            "~class_names",
            {
                0: "triangle",
                1: "square",
                2: "rectangle",
                3: "star",
                4: "rhombus",
                5: "ellipse",
                6: "pentagon",
                7: "four_leaf_clover",
                8: "hexagon",
                9: "circle",
            },
        )

        # ID filter list - only these class IDs will be processed and saved
        self.allowed_class_ids = rospy.get_param("~allowed_class_ids", [1, 2, 8, 5])

        # Create base save directory if it doesn't exist
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
            rospy.loginfo(f"Created base directory: {self.save_directory}")

        # Keep track of created class directories and counters
        self.created_dirs = set()
        self.class_counters = {}

        self.is_active = False

        # Log the allowed class IDs
        rospy.loginfo(f"Allowed class IDs: {self.allowed_class_ids}")
        allowed_names = [
            self.class_names.get(id, f"class_{id}") for id in self.allowed_class_ids
        ]
        rospy.loginfo(f"Allowed class names: {allowed_names}")

        self.service = rospy.Service(
            "/yolo_image_annotator/toggle_annotator", SetBool, self.enable_callback
        )

        # Subscribe to topics using message_filters for synchronization
        image_sub = message_filters.Subscriber("cam_bottom/image_raw", Image)
        yolo_sub = message_filters.Subscriber("yolo_result_bottom", YoloResult)

        # Synchronize messages with a time tolerance
        ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, yolo_sub], queue_size=10, slop=0.1  # 100ms tolerance
        )
        ts.registerCallback(self.synchronized_callback)

        rospy.loginfo("YOLO Image Annotator node started")
        rospy.loginfo(f"Saving images to: {self.save_directory}")

    def enable_callback(self, req):
        """Service callback to enable/disable the annotator"""
        self.is_active = req.data

        if self.is_active:
            rospy.loginfo("YOLO Image Annotator ENABLED")
            message = "Image annotation and saving is now active"
        else:
            rospy.loginfo("YOLO Image Annotator DISABLED")
            message = "Image annotation and saving is now disabled"

        return SetBoolResponse(success=True, message=message)

    def parse_classes(self, yolo_result):
        """Extract (id, name) tuples from YoloResult message, filtering by allowed IDs"""
        classes = []
        for detection in yolo_result.detections.detections:
            for result in detection.results:
                class_id = result.id
                if class_id in self.allowed_class_ids:
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    classes.append((class_id, class_name))
        return list(set(classes)) if classes else []

    def create_class_directory(self, class_name):
        """Create directory for specific class if it doesn't exist"""
        class_dir = os.path.join(self.save_directory, class_name)

        if class_name not in self.created_dirs:
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
                rospy.loginfo(f"Created class directory: {class_dir}")
            self.created_dirs.add(class_name)

        return class_dir

    def draw_class_names(self, image, class_items):
        """Draw class IDs and names on the image"""
        y_offset = 30
        for class_id, class_name in class_items:
            text = f"{class_id}: {class_name}"
            (text_width, text_height), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                self.font_thickness,
            )

            # Draw background rectangle
            cv2.rectangle(
                image,
                (10, y_offset - text_height - 5),
                (10 + text_width, y_offset + baseline + 5),
                (0, 0, 0),  # Black background
                -1,  # Filled rectangle
            )

            # Draw text
            cv2.putText(
                image,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 255, 0),  # Green text
                self.font_thickness,
            )

            y_offset += text_height + 15

        return image

    def synchronized_callback(self, image_msg, yolo_msg):
        """Callback function for synchronized messages"""
        try:

            if not self.is_active:
                return

            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # Check if there are any detections
            if not yolo_msg.detections.detections:
                rospy.logwarn("No detections in YOLO message")
                return

            # Parse (id, name) pairs
            classes = self.parse_classes(yolo_msg)

            # If no allowed classes detected, skip processing
            if not classes:
                rospy.logdebug("No allowed classes detected in current frame")
                return

            # Draw IDs + class names on image
            annotated_image = self.draw_class_names(cv_image.copy(), classes)

            # Save image to each detected class directory
            for class_id, class_name in classes:
                class_dir = self.create_class_directory(class_name)

                if class_name not in self.class_counters:
                    self.class_counters[class_name] = 0
                self.class_counters[class_name] += 1

                # Filename with ID + name
                filename = (
                    f"{class_id}_{class_name}_{self.class_counters[class_name]}.jpg"
                )
                filepath = os.path.join(class_dir, filename)

                # Save the annotated image
                success = cv2.imwrite(filepath, annotated_image)

                if success:
                    rospy.loginfo(f"Saved image: {filepath}")
                else:
                    rospy.logerr(f"Failed to save image: {filepath}")

        except Exception as e:
            rospy.logerr(f"Error processing messages: {str(e)}")

    def run(self):
        """Keep the node running"""
        rospy.spin()


if __name__ == "__main__":
    try:
        annotator = YoloImageAnnotator()
        annotator.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO Image Annotator node terminated")
    except Exception as e:
        rospy.logerr(f"Error starting node: {str(e)}")

#!/usr/bin/env python3

import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from ultralytics_ros.msg import YoloResult
from nav_msgs.msg import Odometry

class ObjectPositionEstimator:
    def __init__(self):
        rospy.init_node('object_position_estimator', anonymous=True)
        
        # Camera parameters
        self.hfov = math.radians(38)  # Horizontal field of view in radians
        self.vfov = math.radians(28)  # Vertical field of view in radians
        self.altitude = None

        rospy.loginfo("Waiting for set_object_transform service...")
        self.set_object_transform_service = rospy.ServiceProxy('/taluy/map/set_object_transform', SetObjectTransform)
        self.set_object_transform_service.wait_for_service()

        self.id_tf_map = {
            9: "bin/whole",
            10: "bin/red",
            11: "bin/blue"
        }

        # Subscriptions
        self.yolo_sub = rospy.Subscriber('/yolo_result', YoloResult, self.yolo_callback)
        self.altitude_sub = rospy.Subscriber('/taluy/sensors/dvl/altitude', Float32, self.altitude_callback)
        
        self.camera_width = 640
        self.camera_height = 480
        rospy.loginfo("Object position estimator node initialized")

    def altitude_callback(self, msg):
        self.altitude = msg.data

    def send_transform(self, transform: TransformStamped):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(f"Failed to set object transform, reason: {resp.message}")

    def yolo_callback(self, msg):
        if self.camera_width is None or self.camera_height is None or self.altitude is None:
            return
        
        for detection in msg.detections.detections:
            if len(detection.results) == 0:
                continue

            detection_id = detection.results[0].id

            if detection_id not in self.id_tf_map:
                continue
            
            # Calculate bounding box center
            box_center_x = detection.bbox.center.x
            box_center_y = detection.bbox.center.y
            
            # Normalize center position to [-0.5, 0.5]
            norm_center_x = (box_center_x - self.camera_width / 2) / self.camera_width
            norm_center_y = (box_center_y - self.camera_height / 2) / self.camera_height
            
            # Calculate the angles based on the camera FOV
            angle_x = norm_center_x * self.hfov
            angle_y = norm_center_y * self.vfov
            
            # Calculate the offset in the bottom_camera_optical_link frame
            offset_x = math.tan(angle_x) * self.altitude * -1.0
            offset_y = math.tan(angle_y) * self.altitude * -1.0
            
            transform_message = TransformStamped()
            transform_message.header.stamp = rospy.Time.now()
            transform_message.header.frame_id = "taluy/base_link/bottom_camera_optical_link"
            transform_message.child_frame_id = f"{self.id_tf_map[detection_id]}_link"
            transform_message.transform.translation.x = offset_x
            transform_message.transform.translation.y = offset_y
            transform_message.transform.translation.z = self.altitude
            transform_message.transform.rotation.x = 0.0
            transform_message.transform.rotation.y = 0.0
            transform_message.transform.rotation.z = 0.0
            transform_message.transform.rotation.w = 1.0

            self.send_transform(transform_message)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ObjectPositionEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass

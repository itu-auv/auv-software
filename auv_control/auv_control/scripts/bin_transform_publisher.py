#!/usr/bin/env python

import rospy
import math
import tf
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float32
from ultralytics_ros.msg import YoloResult
from nav_msgs.msg import Odometry

class ObjectPositionEstimator:
    def __init__(self):
        rospy.init_node('object_position_estimator', anonymous=True)
        
        # Camera parameters
        self.hfov = math.radians(38)  # Horizontal field of view in radians
        self.vfov = math.radians(28)  # Vertical field of view in radians
        self.altitude = None
        
        # Subscriptions
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.yolo_sub = rospy.Subscriber('/yolo_result', YoloResult, self.yolo_callback)
        self.altitude_sub = rospy.Subscriber('/taluy/sensors/dvl/altitude', Float32, self.altitude_callback)
        
        # Transform broadcaster and listener
        self.br = tf.TransformBroadcaster()
        self.listener = tf.TransformListener()
        
        self.camera_width = 1280
        self.camera_height = 720
        self.odom_pose = None

    def odom_callback(self, msg):
        self.odom_pose = msg.pose.pose

    def altitude_callback(self, msg):
        self.altitude = msg.data

    def yolo_callback(self, msg):
        if self.camera_width is None or self.camera_height is None or self.odom_pose is None or self.altitude is None:
            return
        
        for detection in msg.detections.detections:
            if detection.results[0].id != 0:
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
            
            # Calculate the offset in the base_link frame
            offset_x = math.tan(angle_x) * self.altitude * -1
            offset_y = math.tan(angle_y) * self.altitude
            
            # Create a PoseStamped message for the object in the base_link frame
            object_pose_base_link = PoseStamped()
            object_pose_base_link.header.stamp = rospy.Time.now()
            object_pose_base_link.header.frame_id = "base_link"
            object_pose_base_link.pose.position.x = offset_x
            object_pose_base_link.pose.position.y = offset_y
            object_pose_base_link.pose.position.z = 0.0
            object_pose_base_link.pose.orientation.x = 0.0
            object_pose_base_link.pose.orientation.y = 0.0
            object_pose_base_link.pose.orientation.z = 0.0
            object_pose_base_link.pose.orientation.w = 1.0

            # Use the tf listener to transform the object position from base_link to odom frame
            try:
                self.listener.waitForTransform("odom", "base_link", rospy.Time(0), rospy.Duration(1.0))
                transformed_position = self.listener.transformPose("odom", object_pose_base_link)
                
                # Broadcast the transformed position
                self.br.sendTransform(
                    (transformed_position.pose.position.x, 
                     transformed_position.pose.position.y, 
                     transformed_position.pose.position.z),
                    (transformed_position.pose.orientation.x, 
                     transformed_position.pose.orientation.y, 
                     transformed_position.pose.orientation.z, 
                     transformed_position.pose.orientation.w),
                    rospy.Time.now(),
                    "object",
                    "odom"
                )
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ObjectPositionEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass

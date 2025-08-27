#!/usr/bin/env python3

import rospy
import tf2_ros
import tf.transformations
import numpy as np
from geometry_msgs.msg import Pose, TransformStamped, Vector3, Quaternion
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class PipelineTransformServiceNode:
    def __init__(self):
        self.is_enabled = False
        rospy.init_node("pipeline_trajectory_publisher_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Service to broadcast transforms
        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        # Parameters
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.robot_base_frame = rospy.get_param("~robot_base_frame", "taluy/base_link")

        # Frame names for pipeline corners
        self.pipe_corner_frames = [
            "pipe_corner_1",
            "pipe_corner_2",
            "pipe_corner_3",
            "pipe_corner_4",
            "pipe_corner_5",
            "pipe_corner_6",
            "pipe_corner_7",
            "pipe_corner_8",
            "pipe_corner_9",
        ]

        # Service to enable/disable pipeline trajectory generation
        self.set_enable_service = rospy.Service(
            "toggle_pipeline_trajectory", SetBool, self.handle_enable_service
        )

        rospy.loginfo("Pipeline trajectory publisher node initialized")

    def handle_enable_service(self, request):
        """Handle enable/disable service requests"""
        self.is_enabled = request.data
        message = (
            f"Pipeline trajectory transform publishing is set to: {self.is_enabled}"
        )
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def get_pose_from_transform(self, transform: TransformStamped) -> Pose:
        """Convert TransformStamped to Pose"""
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        """Build TransformStamped message"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform: TransformStamped):
        """Send transform via service"""
        req = SetObjectTransformRequest()
        req.transform = transform
        try:
            resp = self.set_object_transform_service.call(req)
            if not resp.success:
                rospy.logwarn(
                    f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
                )
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def create_relative_pose(
        self, base_pose: Pose, forward: float, left: float, up: float = 0.0
    ) -> Pose:
        """
        Create a new pose relative to base pose
        forward: positive values move forward in robot's direction
        left: positive values move left relative to robot's orientation
        up: positive values move up
        """
        # Get base orientation as rotation matrix
        base_quat = [
            base_pose.orientation.x,
            base_pose.orientation.y,
            base_pose.orientation.z,
            base_pose.orientation.w,
        ]
        base_matrix = tf.transformations.quaternion_matrix(base_quat)

        # Create relative translation vector in robot's frame
        relative_translation = np.array([forward, left, up, 1.0])

        # Transform to world frame
        world_translation = np.dot(base_matrix, relative_translation)

        # Create new pose
        new_pose = Pose()
        new_pose.position.x = base_pose.position.x + world_translation[0]
        new_pose.position.y = base_pose.position.y + world_translation[1]
        new_pose.position.z = base_pose.position.z + world_translation[2]

        # Keep same orientation as robot
        new_pose.orientation = base_pose.orientation

        return new_pose

    def create_pipeline_frames(self):
        """Create all pipeline corner frames based on robot's current position"""
        try:
            # Get robot's current transform
            robot_transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.robot_base_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )

            robot_pose = self.get_pose_from_transform(robot_transform)
            rospy.loginfo(
                f"Robot position: x={robot_pose.position.x:.2f}, y={robot_pose.position.y:.2f}, z={robot_pose.position.z:.2f}"
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"Failed to lookup robot transform: {e}")
            return

        # Create pipeline corner frames according to the specification:
        # pipe_corner_1: 5m forward from robot
        # pipe_corner_2: 2.5m left from pipe_corner_1
        # pipe_corner_3: 2m backward from pipe_corner_2
        # pipe_corner_4: 1.5m right from pipe_corner_3
        # pipe_corner_5: 1.5m backward from pipe_corner_4
        # pipe_corner_6: 1m left from pipe_corner_5
        # pipe_corner_7: 1.5m backward from pipe_corner_6
        # pipe_corner_8: 3m left from pipe_corner_7
        # pipe_corner_9: 5m forward from pipe_corner_8

        # pipe_corner_1: 5 meters forward
        corner_1_pose = self.create_relative_pose(robot_pose, forward=5.0, left=0.0)

        # pipe_corner_2: 2.5 meters left from corner_1
        corner_2_pose = self.create_relative_pose(corner_1_pose, forward=0.0, left=2.5)

        # pipe_corner_3: 2 meters backward from corner_2
        corner_3_pose = self.create_relative_pose(corner_2_pose, forward=-2.0, left=0.0)

        # pipe_corner_4: 1.5 meters right from corner_3
        corner_4_pose = self.create_relative_pose(corner_3_pose, forward=0.0, left=-1.5)

        # pipe_corner_5: 1.5 meters backward from corner_4
        corner_5_pose = self.create_relative_pose(corner_4_pose, forward=-1.5, left=0.0)

        # pipe_corner_6: 1 meter left from corner_5
        corner_6_pose = self.create_relative_pose(corner_5_pose, forward=0.0, left=1.0)

        # pipe_corner_7: 1.5 meters backward from corner_6
        corner_7_pose = self.create_relative_pose(corner_6_pose, forward=-1.5, left=0.0)

        # pipe_corner_8: 3 meters left from corner_7
        corner_8_pose = self.create_relative_pose(corner_7_pose, forward=0.0, left=3.0)

        # pipe_corner_9: 5 meters forward from corner_8
        corner_9_pose = self.create_relative_pose(corner_8_pose, forward=5.0, left=0.0)

        # Store all poses
        corner_poses = [
            corner_1_pose,
            corner_2_pose,
            corner_3_pose,
            corner_4_pose,
            corner_5_pose,
            corner_6_pose,
            corner_7_pose,
            corner_8_pose,
            corner_9_pose,
        ]

        # Send all transforms
        for i, pose in enumerate(corner_poses):
            frame_name = self.pipe_corner_frames[i]
            transform = self.build_transform_message(frame_name, pose)
            self.send_transform(transform)
            rospy.loginfo(
                f"Created {frame_name} at position: "
                f"x={pose.position.x:.2f}, y={pose.position.y:.2f}, z={pose.position.z:.2f}"
            )

        rospy.loginfo("All pipeline corner frames created successfully")

    def run(self):
        """Main run loop"""
        rospy.loginfo("Pipeline trajectory publisher is running...")
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.is_enabled:
                self.create_pipeline_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = PipelineTransformServiceNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Pipeline trajectory publisher node shutdown")

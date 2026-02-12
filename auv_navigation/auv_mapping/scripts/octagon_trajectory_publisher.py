#!/usr/bin/env python3

import numpy as np
import rospy
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, TransformStamped
from std_srvs.srv import SetBool, SetBoolResponse
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest


class OctagonTransformServiceNode:
    def __init__(self):
        self.enable = False
        rospy.init_node("create_octagon_frame_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.odom_frame = "odom"
        self.robot_frame = "taluy/base_link"
        self.octagon_frame = "octagon_link"

        self.octagon_closer_frame = "octagon_closer_link"
        self.closer_distance = rospy.get_param(
            "~closer_distance", 2.0
        )  # distance from octagon to closer frame
        # Search frames configuration
        self.search_distance = rospy.get_param(
            "~search_distance", 0.7
        )  # distance from octagon to search frames
        self.search_frames = {
            "octagon_search_forward": "forward",  # towards robot (back from octagon)
            "octagon_search_backward": "backward",  # away from robot
            "octagon_search_left": "left",  # perpendicular left
            "octagon_search_right": "right",  # perpendicular right
        }

        self.set_enable_service = rospy.Service(
            "set_transform_octagon_frame", SetBool, self.handle_enable_service
        )

    def get_pose(self, transform: TransformStamped) -> Pose:
        pose = Pose()
        pose.position = transform.transform.translation
        pose.orientation = transform.transform.rotation
        return pose

    def build_transform_message(
        self, child_frame_id: str, pose: Pose
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.odom_frame
        t.child_frame_id = child_frame_id
        t.transform.translation = pose.position
        t.transform.rotation = pose.orientation
        return t

    def send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logwarn(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def create_octagon_frame(self):
        if not self.enable:
            return

        try:
            transform_robot = self.tf_buffer.lookup_transform(
                self.odom_frame, self.robot_frame, rospy.Time(0), rospy.Duration(4.0)
            )
            transform_octagon = self.tf_buffer.lookup_transform(
                self.odom_frame,
                self.octagon_frame,
                rospy.Time.now(),
                rospy.Duration(4.0),
            )

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return

        robot_pose = self.get_pose(transform_robot)
        octagon_pose = self.get_pose(transform_octagon)

        robot_pos = np.array(
            [robot_pose.position.x, robot_pose.position.y, robot_pose.position.z]
        )
        octagon_pos = np.array(
            [octagon_pose.position.x, octagon_pose.position.y, octagon_pose.position.z]
        )

        direction_vector_2d = octagon_pos[:2] - robot_pos[:2]
        total_distance_2d = np.linalg.norm(direction_vector_2d)

        if total_distance_2d == 0:
            rospy.logwarn(
                "Robot and octagon are at the same XY position! Cannot create frame."
            )
            return

        direction_unit_2d = direction_vector_2d / total_distance_2d

        # Calculate yaw from the direction vector (robot to octagon)
        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        orientation = Pose().orientation
        orientation.x = q[0]
        orientation.y = q[1]
        orientation.z = q[2]
        orientation.w = q[3]

        # Calculate position for closer frame
        closer_pos_2d = octagon_pos[:2] - (direction_unit_2d * self.closer_distance)
        closer_pos = np.append(closer_pos_2d, robot_pos[2])

        # Create closer frame
        closer_pose = Pose()
        closer_pose.position.x, closer_pose.position.y, closer_pose.position.z = (
            closer_pos
        )
        closer_pose.orientation = orientation

        # Send the transform
        closer_transform = self.build_transform_message(
            self.octagon_closer_frame, closer_pose
        )
        self.send_transform(closer_transform)

        # Create search frames around octagon
        self.create_search_frames(octagon_pos, direction_unit_2d, robot_pos[2])

    def create_search_frames(self, octagon_pos, direction_unit_2d, z_height):
        """
        Create 4 search frames in a + pattern around octagon.
        - forward: away from robot (further from octagon perspective)
        - backward: towards robot direction
        - left: perpendicular left
        - right: perpendicular right

        All frames have the same orientation as closer_frame (facing robot->octagon direction).
        """
        # Calculate perpendicular vector (90 degrees rotation in XY plane)
        # Rotate direction_unit_2d by 90 degrees: (x, y) -> (-y, x)
        perpendicular_unit_2d = np.array([-direction_unit_2d[1], direction_unit_2d[0]])

        # Calculate yaw from direction vector (same as closer_frame)
        # All frames face forward (robot to octagon direction)
        yaw = np.arctan2(direction_unit_2d[1], direction_unit_2d[0])
        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        for frame_name, direction_type in self.search_frames.items():
            # Calculate position offset based on direction type
            if direction_type == "forward":
                # Away from robot (same as direction vector)
                offset_2d = direction_unit_2d * self.search_distance
            elif direction_type == "backward":
                # Towards robot (opposite of direction vector)
                offset_2d = -direction_unit_2d * self.search_distance
            elif direction_type == "left":
                # Perpendicular left
                offset_2d = perpendicular_unit_2d * self.search_distance
            elif direction_type == "right":
                # Perpendicular right
                offset_2d = -perpendicular_unit_2d * self.search_distance
            else:
                continue

            # Calculate frame position (octagon center + offset)
            frame_pos_2d = octagon_pos[:2] + offset_2d
            frame_pos = np.array([frame_pos_2d[0], frame_pos_2d[1], z_height])

            # Create pose
            search_pose = Pose()
            search_pose.position.x = frame_pos[0]
            search_pose.position.y = frame_pos[1]
            search_pose.position.z = frame_pos[2]
            search_pose.orientation.x = q[0]
            search_pose.orientation.y = q[1]
            search_pose.orientation.z = q[2]
            search_pose.orientation.w = q[3]

            # Send transform
            search_transform = self.build_transform_message(frame_name, search_pose)
            self.send_transform(search_transform)

    def handle_enable_service(self, req: SetBool):
        self.enable = req.data
        rospy.loginfo(
            f"Octagon frame publishing {'enabled' if self.enable else 'disabled'}"
        )
        return SetBoolResponse(success=True)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.create_octagon_frame()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = OctagonTransformServiceNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

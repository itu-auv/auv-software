#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import math
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
from std_srvs.srv import Trigger, TriggerResponse
import tf
import sys


class TransformServiceNode:
    def __init__(self):
        rospy.init_node("create_gate_frames_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.set_object_transform_service = rospy.ServiceProxy(
            "/taluy/map/set_object_transform", SetObjectTransform
        )
        self.set_object_transform_service.wait_for_service()

        self.frame1 = "gate_blue_arrow_link"
        self.frame2 = "gate_red_arrow_link"
        self.new_frame_away = "gate_exit"
        self.new_frame_closer = "gate_enterance"
        self.world_frame = "odom"

        self.selected_frame = "gate_blue_arrow_link"

    def create_new_frames(self):
        try:
            trans1 = self.tf_buffer.lookup_transform(
                self.world_frame, self.frame1, rospy.Time(0), rospy.Duration(1)
            )
            trans2 = self.tf_buffer.lookup_transform(
                self.world_frame, self.frame2, rospy.Time(0), rospy.Duration(1)
            )

            x1, y1, z1 = (
                trans1.transform.translation.x,
                trans1.transform.translation.y,
                trans1.transform.translation.z,
            )
            x2, y2, z2 = (
                trans2.transform.translation.x,
                trans2.transform.translation.y,
                trans2.transform.translation.z,
            )

            if self.selected_frame == self.frame1:
                selected_x, selected_y, selected_z = x1, y1, z1
                other_x, other_y, other_z = x2, y2, z2
            elif self.selected_frame == self.frame2:
                selected_x, selected_y, selected_z = x2, y2, z2
                other_x, other_y, other_z = x1, y1, z1
            else:
                rospy.logerr("Invalid selected frame")
                return

            # Calculate the vector perpendicular to the line connecting selected_frame and the other frame
            dx = other_x - selected_x
            dy = other_y - selected_y
            length = math.sqrt(dx**2 + dy**2)
            perp_dx = -dy / length
            perp_dy = dx / length

            # New frame 1 meter away
            new_x_away = selected_x + perp_dx
            new_y_away = selected_y + perp_dy

            # New frame 1 meter closer
            new_x_closer = selected_x - perp_dx * 1.7
            new_y_closer = selected_y - perp_dy * 1.7

            # Calculate orientation to look towards the selected frame
            angle_away = math.atan2(selected_y - new_y_away, selected_x - new_x_away)
            angle_closer = math.atan2(
                selected_y - new_y_closer, selected_x - new_x_closer
            )
            quat_away = tf.transformations.quaternion_from_euler(0, 0, angle_away)
            quat_closer = tf.transformations.quaternion_from_euler(0, 0, angle_closer)

            # Calculate distances from new frames to odom
            distance_away = math.sqrt(new_x_away**2 + new_y_away**2)
            distance_closer = math.sqrt(new_x_closer**2 + new_y_closer**2)

            # Determine which frame is closer to odom
            if distance_closer < distance_away:
                entrance_x, entrance_y, entrance_z = (
                    new_x_closer,
                    new_y_closer,
                    selected_z,
                )
                entrance_quat = quat_closer
                exit_x, exit_y, exit_z = new_x_away, new_y_away, selected_z
                exit_quat = quat_away
            else:
                entrance_x, entrance_y, entrance_z = new_x_away, new_y_away, selected_z
                entrance_quat = quat_away
                exit_x, exit_y, exit_z = new_x_closer, new_y_closer, selected_z
                exit_quat = quat_closer

            # Create TransformStamped messages for entrance and exit
            t_entrance = geometry_msgs.msg.TransformStamped()
            t_entrance.header.stamp = rospy.Time.now()
            t_entrance.header.frame_id = self.world_frame
            t_entrance.child_frame_id = self.new_frame_closer
            t_entrance.transform.translation.x = entrance_x
            t_entrance.transform.translation.y = entrance_y
            t_entrance.transform.translation.z = entrance_z
            t_entrance.transform.rotation.x = entrance_quat[0]
            t_entrance.transform.rotation.y = entrance_quat[1]
            t_entrance.transform.rotation.z = entrance_quat[2]
            t_entrance.transform.rotation.w = entrance_quat[3]

            t_exit = geometry_msgs.msg.TransformStamped()
            t_exit.header.stamp = rospy.Time.now()
            t_exit.header.frame_id = self.world_frame
            t_exit.child_frame_id = self.new_frame_away
            t_exit.transform.translation.x = exit_x
            t_exit.transform.translation.y = exit_y
            t_exit.transform.translation.z = exit_z
            t_exit.transform.rotation.x = entrance_quat[0]
            t_exit.transform.rotation.y = entrance_quat[1]
            t_exit.transform.rotation.z = entrance_quat[2]
            t_exit.transform.rotation.w = entrance_quat[3]

            # Send transforms using the service
            self.send_transform(t_entrance)
            self.send_transform(t_exit)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"TF lookup failed: {e}")

    def send_transform(self, transform):
        req = SetObjectTransformRequest()
        req.transform = transform
        resp = self.set_object_transform_service.call(req)
        if not resp.success:
            rospy.logerr(
                f"Failed to set transform for {transform.child_frame_id}: {resp.message}"
            )

    def spin(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            self.create_new_frames()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = TransformServiceNode()
        node.spin()  # Keep the node running
    except rospy.ROSInterruptException:
        pass

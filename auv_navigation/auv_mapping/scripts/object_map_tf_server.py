#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import threading
from geometry_msgs.msg import TransformStamped
from auv_msgs.srv import SetObjectTransform, SetObjectTransformResponse


class ObjectMapTFServer:
    def __init__(self):
        rospy.init_node("object_map_tf_server", anonymous=True)

        # Load parameters
        self.static_frame = rospy.get_param("~static_frame", "odom")
        self.update_rate = rospy.get_param("~rate", 10.0)  # Hz

        # Transform storage
        self.transforms = {}  # Dictionary to store target frames and their transforms

        # Lock for thread-safe access to transforms
        self.lock = threading.Lock()

        # ROS service to set object transform
        self.service = rospy.Service(
            "set_object_transform", SetObjectTransform, self.handle_set_transform
        )

        # Transform broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Transform listener to get odom->base_link transform
        self.tf_listener = tf.TransformListener()

        rospy.loginfo(
            "ObjectMapTFServer initialized. Static frame: %s", self.static_frame
        )

    def handle_set_transform(self, req):
        parent_frame = req.transform.header.frame_id
        target_frame = req.transform.child_frame_id

        try:
            # Wait for the transform from static_frame (odom) to parent_frame (base_link)
            self.tf_listener.waitForTransform(
                self.static_frame, parent_frame, rospy.Time(0), rospy.Duration(4.0)
            )
            (trans, rot) = self.tf_listener.lookupTransform(
                self.static_frame, parent_frame, rospy.Time(0)
            )

            # Convert the provided transform to the static frame
            static_transform = TransformStamped()
            static_transform.header.stamp = rospy.Time.now()
            static_transform.header.frame_id = self.static_frame
            static_transform.child_frame_id = target_frame

            # Parent to target transformation matrix
            parent_to_target = tf.transformations.quaternion_matrix(
                (
                    req.transform.transform.rotation.x,
                    req.transform.transform.rotation.y,
                    req.transform.transform.rotation.z,
                    req.transform.transform.rotation.w,
                )
            )
            parent_to_target[0:3, 3] = [
                req.transform.transform.translation.x,
                req.transform.transform.translation.y,
                req.transform.transform.translation.z,
            ]

            # Static frame to parent transformation matrix
            static_to_parent = tf.transformations.quaternion_matrix(rot)
            static_to_parent[0:3, 3] = trans

            # Combined transformation: static frame to target
            static_to_target = tf.transformations.concatenate_matrices(
                static_to_parent, parent_to_target
            )

            # Extract translation and rotation
            static_translation = tf.transformations.translation_from_matrix(
                static_to_target
            )
            static_rotation = tf.transformations.quaternion_from_matrix(
                static_to_target
            )

            static_transform.transform.translation.x = static_translation[0]
            static_transform.transform.translation.y = static_translation[1]
            static_transform.transform.translation.z = static_translation[2]
            static_transform.transform.rotation.x = static_rotation[0]
            static_transform.transform.rotation.y = static_rotation[1]
            static_transform.transform.rotation.z = static_rotation[2]
            static_transform.transform.rotation.w = static_rotation[3]

            # Store or update the transform in the dictionary using the lock
            with self.lock:
                self.transforms[target_frame] = static_transform
            
            rospy.loginfo("Stored static transform for frame: %s", target_frame)

            return SetObjectTransformResponse(
                success=True, message=f"Stored transform for frame: {target_frame}"
            )

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
        ) as e:
            rospy.logerr("Error occurred while looking up transform: %s", str(e))
            return SetObjectTransformResponse(
                success=False, message=f"Failed to capture transform: {str(e)}"
            )

    def publish_transforms(self):
        rate = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown():
            with self.lock:
                for transform in self.transforms.values():
                    transform.header.stamp = rospy.Time.now()
                    self.tf_broadcaster.sendTransform(transform)
            rate.sleep()


if __name__ == "__main__":
    try:
        server = ObjectMapTFServer()
        server.publish_transforms()
    except rospy.ROSInterruptException:
        pass

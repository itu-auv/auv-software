#!/usr/bin/env python3

import rospy
import math
import tf
import message_filters
from geometry_msgs.msg import (
    TransformStamped,
    PointStamped,
)
from auv_msgs.srv import SetObjectTransform, SetObjectTransformRequest
import tf2_ros
import tf2_geometry_msgs
import copy
import threading
from std_srvs.srv import SetBool, SetBoolResponse


class Scene:
    def __init__(self):
        self.lock = threading.Lock()
        self.objects = {}

    def add_object_to_location(self, id: int, x: float, y: float, z: float):
        with self.lock:
            if id not in self.objects:
                self.objects[id] = []

            # check if the new location belongs to already existing object or a new object
            for obj in self.objects[id]:
                distance = math.sqrt(
                    (obj.x - x) ** 2 + (obj.y - y) ** 2 + (obj.z - z) ** 2
                )
                if distance < 4.0:
                    obj.update_position(x, y, z)
                    return

            # if it is octagon, dont add new octagon if there is already one
            if id == 14 and len(self.objects[id]) > 0:
                pass
            else:
                new_object = SceneObject(id, x, y, z)
                self.objects[id].append(new_object)

    def get_objects(self):
        with self.lock:
            return copy.deepcopy(self.objects)

    def update_objects(self):
        with self.lock:
            for key, value in self.objects.items():
                for obj in value:
                    obj.update_filtered_position()


class SceneObject:
    def __init__(self, id: int, x: float, y: float, z: float):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.filtered_x = x
        self.filtered_y = y
        self.filtered_z = z

    def get_position(self):
        return self.x, self.y, self.z

    def update_filtered_position(self):
        alpha = 0.2  # Weight for new measurements (80-20 filter)
        self.filtered_x = alpha * self.x + (1 - alpha) * self.filtered_x
        self.filtered_y = alpha * self.y + (1 - alpha) * self.filtered_y
        self.filtered_z = alpha * self.z + (1 - alpha) * self.filtered_z

    def update_position(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


id_to_name_map = {
    8: "red_buoy",
    4: "gate_red_arrow",
    3: "gate_blue_arrow",
    5: "gate_middle_part",
    12: "torpedo_map",
    9: "bin_whole",
    14: "octagon",
}


class MappingNode:
    def __init__(self):
        rospy.loginfo("Initializing mapping node")
        rospy.init_node("mapping_node", anonymous=True)

        self.scene = Scene()

        # Initialize TransformBroadcaster
        self.broadcaster = tf2_ros.TransformBroadcaster()

        # Initialize tf2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        # Create subscribers for each detection ID
        self.point_subscribers = {}
        for obj_id in id_to_name_map:
            topic = f"/detection/{obj_id}/point"
            # Using regular ROS subscriber instead of message_filters
            sub = rospy.Subscriber(
                topic, PointStamped, self.points_callback, callback_args=obj_id
            )
            self.point_subscribers[obj_id] = sub

        # Timer for publishing transforms
        self.transform_timer = rospy.Timer(
            rospy.Duration(0.1), self.scene_transform_publisher_callback
        )

        rospy.loginfo("Mapping node initialized")

        self.enable_mapping = True
        self.set_enable_service = rospy.Service(
            "set_mapping_enable", SetBool, self.handle_enable_mapping
        )

    def transform_point_to_odom(self, point_msg: PointStamped) -> PointStamped:
        try:
            # Try to get the latest transform available
            transform = self.tf_buffer.lookup_transform(
                "odom",
                point_msg.header.frame_id,
                rospy.Time(0),  # get the latest transform
                rospy.Duration(1.0),
            )
            point_odom = tf2_geometry_msgs.do_transform_point(point_msg, transform)
            return point_odom
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn(f"Transform failed: {str(e)}")

    def points_callback(self, point_msg, detection_id):
        if not self.enable_mapping:
            return

        if point_msg is None:
            rospy.logwarn(
                f"Received None point message for detection ID: {detection_id}"
            )
            return

        # Transform point to odom frame
        point_odom = self.transform_point_to_odom(point_msg)
        if point_odom is None:
            rospy.logwarn(f"Failed to transform point for detection ID {detection_id}")
            return

        # Add object to scene
        self.scene.add_object_to_location(
            detection_id, point_odom.point.x, point_odom.point.y, point_odom.point.z
        )
        rospy.loginfo(f"Detection ID: {detection_id}")

        # Immediately publish transforms after adding a new object

    def scene_transform_publisher_callback(self, event):
        if not self.enable_mapping:
            return
        self.scene.update_objects()

        objects = self.scene.get_objects()

        for obj_id, obj_list in objects.items():
            if obj_id not in id_to_name_map:
                rospy.logwarn(f"Object ID {obj_id} not found in id_to_name_map")
                continue

            obj_name = id_to_name_map[obj_id]

            # Find the closest object to taluy/base_link
            closest_object = None
            min_distance = float("inf")

            for obj in obj_list:
                point = PointStamped()
                point.header.frame_id = "odom"
                point.point.x = obj.filtered_x
                point.point.y = obj.filtered_y
                point.point.z = obj.filtered_z

                try:
                    transform = self.tf_buffer.lookup_transform(
                        "taluy/base_link/front_camera_optical_link",
                        "odom",
                        rospy.Time(0),
                        rospy.Duration(1.0),
                    )
                    point_base = tf2_geometry_msgs.do_transform_point(point, transform)

                    distance = math.sqrt(
                        point_base.point.x**2
                        + point_base.point.y**2
                        + point_base.point.z**2
                    )

                    if distance < min_distance:
                        min_distance = distance
                        closest_object = obj

                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ) as e:
                    rospy.logerr(f"Error looking up transform: {e}")
                    continue
            current_time = rospy.Time.now()
            # Publish transforms for all objects
            for obj in obj_list:
                transform = TransformStamped()
                transform.header.stamp = current_time
                transform.header.frame_id = "odom"

                # Set child frame ID based on whether it's the closest object
                if obj == closest_object:
                    transform.child_frame_id = f"{obj_name}_link"
                else:
                    transform.child_frame_id = f"{obj_name}_{obj_list.index(obj)}_link"

                # Set translation
                transform.transform.translation.x = obj.filtered_x
                transform.transform.translation.y = obj.filtered_y
                transform.transform.translation.z = obj.filtered_z

                # Set rotation (identity quaternion)
                transform.transform.rotation.w = 1.0
                transform.transform.rotation.x = 0.0
                transform.transform.rotation.y = 0.0
                transform.transform.rotation.z = 0.0

                # Broadcast transform
                self.broadcaster.sendTransform(transform)
        # Update filtered positions
    def handle_enable_mapping(self, req):
        self.enable_mapping = req.data
        message = f"Mapping node enable : {self.enable_mapping}"
        rospy.loginfo(message)
        return SetBoolResponse(success=True, message=message)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        node = MappingNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

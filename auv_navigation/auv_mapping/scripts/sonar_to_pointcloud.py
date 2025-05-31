#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Range, PointCloud2
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from std_msgs.msg import Header, Empty
from std_srvs.srv import SetBool, SetBoolResponse
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs import point_cloud2
import math
import numpy as np
from sklearn.cluster import DBSCAN
import threading


class SonarToPointCloud:
    def __init__(self):
        rospy.init_node("sonar_to_pointcloud", anonymous=True)

        self.max_point_cloud_size = rospy.get_param("~max_point_cloud_size", 200000)
        self.min_valid_range = rospy.get_param("~min_valid_range", 0.3)
        self.max_valid_range = rospy.get_param("~max_valid_range", 100.0)
        self.frame_id = rospy.get_param("~frame_id", "odom")
        self.sonar_front_frame = rospy.get_param(
            "~sonar_front_frame", "taluy/base_link/sonar_front_link"
        )
        self.sonar_right_frame = rospy.get_param(
            "~sonar_right_frame", "taluy/base_link/sonar_right_link"
        )
        self.sonar_left_frame = rospy.get_param(
            "~sonar_left_frame", "taluy/base_link/sonar_left_link"
        )

        self.services_active = False
        self.lock = threading.RLock()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.point_cloud_data = []
        self.reference_z = None
        self.point_count_since_last_detection = 0

        self.pc_publisher = rospy.Publisher(
            "/sonar/point_cloud", PointCloud2, queue_size=10
        )

        self.start_service = rospy.Service(
            "/sonar/start_mapping", SetBool, self.handle_start_service
        )
        self.clear_service = rospy.Service(
            "/sonar/clear_points", SetBool, self.handle_clear_service
        )

        self.sonar_front_subscriber = None
        self.sonar_right_subscriber = None
        self.sonar_left_subscriber = None

    def handle_start_service(self, req):
        with self.lock:
            if req.data:
                if not self.services_active:
                    self.services_active = True
                    self.sonar_front_subscriber = rospy.Subscriber(
                        "/taluy/sensors/sonar_front/data",
                        Range,
                        self.sonar_front_callback,
                    )
                    self.sonar_right_subscriber = rospy.Subscriber(
                        "/taluy/sensors/sonar_right/data",
                        Range,
                        self.sonar_right_callback,
                    )
                    self.sonar_left_subscriber = rospy.Subscriber(
                        "/taluy/sensors/sonar_left/data",
                        Range,
                        self.sonar_left_callback,
                    )
                    rospy.loginfo(
                        "Pool boundary detection started with all three sonars"
                    )
                    return SetBoolResponse(success=True, message="Service started")
                else:
                    return SetBoolResponse(
                        success=False, message="Service is already running"
                    )
            else:
                if self.services_active:
                    self.services_active = False
                    if self.sonar_front_subscriber:
                        self.sonar_front_subscriber.unregister()
                        self.sonar_front_subscriber = None
                    if self.sonar_right_subscriber:
                        self.sonar_right_subscriber.unregister()
                        self.sonar_right_subscriber = None
                    if self.sonar_left_subscriber:
                        self.sonar_left_subscriber.unregister()
                        self.sonar_left_subscriber = None
                    rospy.loginfo("Pool boundary detection stopped")
                    return SetBoolResponse(success=True, message="Service stopped")
                else:
                    return SetBoolResponse(
                        success=False, message="Service is already stopped"
                    )

    def handle_clear_service(self, req):
        with self.lock:
            if req.data:
                self.point_cloud_data = []
                self.reference_z = None
                self.point_count_since_last_detection = 0
                rospy.loginfo("Point cloud data cleared")
                self.publish_point_cloud()
                self.publish_boundaries()
                return SetBoolResponse(success=True, message="Point cloud cleared")
            return SetBoolResponse(success=False, message="No action taken")

    def get_transform(self, target_frame, source_frame, timestamp):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, timestamp, timeout=rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(10, f"Transform error: {str(e)}")
            return None

    def transform_point_to_odom(self, point_stamped):
        transform = self.get_transform(
            self.frame_id, point_stamped.header.frame_id, point_stamped.header.stamp
        )

        if transform:
            return tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return None

    def process_sonar_reading(self, msg, sonar_frame):
        if not self.services_active:
            return

        if not (self.min_valid_range < msg.range < self.max_valid_range):
            return

        sensor_point = PointStamped()
        sensor_point.header.stamp = msg.header.stamp
        sensor_point.header.frame_id = sonar_frame
        sensor_point.point.x = msg.range
        sensor_point.point.y = 0.0
        sensor_point.point.z = 0.0

        odom_point = self.transform_point_to_odom(sensor_point)
        if not odom_point:
            return

        if self.reference_z is None:
            self.reference_z = odom_point.point.z
            rospy.loginfo(f"Reference Z set to: {self.reference_z}")

        if len(self.point_cloud_data) >= self.max_point_cloud_size:
            self.point_cloud_data.pop(0)

        self.point_cloud_data.append(
            [odom_point.point.x, odom_point.point.y, odom_point.point.z]
        )

        self.point_count_since_last_detection += 1

        self.publish_point_cloud()

    def sonar_front_callback(self, msg):
        with self.lock:
            self.process_sonar_reading(msg, self.sonar_front_frame)

    def sonar_right_callback(self, msg):
        with self.lock:
            self.process_sonar_reading(msg, self.sonar_right_frame)

    def sonar_left_callback(self, msg):
        with self.lock:
            self.process_sonar_reading(msg, self.sonar_left_frame)

    def publish_point_cloud(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        pc2 = point_cloud2.create_cloud_xyz32(header, self.point_cloud_data)
        self.pc_publisher.publish(pc2)

    def run(self):
        rospy.loginfo(
            "Sonar to pointcloud node initialized. Use services to start/stop."
        )
        rospy.spin()


if __name__ == "__main__":
    try:
        sonar_to_pointcloud = SonarToPointCloud()
        sonar_to_pointcloud.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node ended!")

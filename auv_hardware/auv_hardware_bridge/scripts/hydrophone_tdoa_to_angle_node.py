#!/usr/bin/env python3

import numpy as np
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Int16MultiArray
from visualization_msgs.msg import Marker


C = 1504.34
SAMPLE_RATE = 250000
W = 0.65
L = 0.35
SENSORS = np.array(
    [
        [0.00, 0.00],
        [0.00, L],
        [W, L],
        [W, 0.00],
    ]
)
A = SENSORS[1:] - SENSORS[0]
ROBOT_YAW_OFFSET_DEG = 90.0


def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def calculate_angle(tdoa_samples):
    tdoas = np.array(tdoa_samples, dtype=float) / float(SAMPLE_RATE)
    b_vec = C * tdoas
    k_est, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    k_est = -k_est

    norm_k = np.linalg.norm(k_est)
    if norm_k > 1e-12:
        k_est /= norm_k
    else:
        k_est = np.array([0.0, 0.0])

    return np.arctan2(k_est[1], k_est[0])


def to_robot_angle(hydrophone_angle, yaw_offset):
    return normalize_angle(hydrophone_angle + yaw_offset)


class HydrophoneTDOAToAngle:
    def __init__(self):
        rospy.init_node("hydrophone_tdoa_to_angle_node", anonymous=True)
        yaw_offset_deg = rospy.get_param("~yaw_offset_deg", ROBOT_YAW_OFFSET_DEG)
        self.yaw_offset = np.radians(yaw_offset_deg)
        self.marker_length = rospy.get_param("~marker_length", 1.5)
        self.base_link = rospy.get_param("~base_link", "taluy/base_link")
        self.marker_pub = rospy.Publisher(
            "acoustic/hydrophone/marker", Marker, queue_size=10
        )
        self.angle_pub = rospy.Publisher(
            "acoustic/hydrophone/base_angle", Float32, queue_size=10
        )
        self.tdoa_sub = rospy.Subscriber(
            "acoustic/hydrophone/tdoa", Int16MultiArray, self.tdoa_callback
        )

    def tdoa_callback(self, msg):
        if len(msg.data) < 4:
            rospy.logwarn("Incomplete TDOA data: %s", list(msg.data))
            return

        hydrophone_angle = calculate_angle(msg.data[0:3])
        robot_angle = to_robot_angle(hydrophone_angle, self.yaw_offset)

        self.angle_pub.publish(Float32(data=float(robot_angle)))

        marker = self.make_marker(robot_angle, msg.data[3])
        self.marker_pub.publish(marker)

    def make_marker(self, robot_angle, magnitude):
        end_x = np.cos(robot_angle) * self.marker_length
        end_y = np.sin(robot_angle) * self.marker_length

        marker = Marker()
        marker.header.frame_id = self.base_link
        marker.header.stamp = rospy.Time.now()
        marker.ns = "tdoa_direction"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.points = [
            Point(0.0, 0.0, 0.0),
            Point(float(end_x), float(end_y), 0.0),
        ]
        marker.scale.x = 0.05
        marker.scale.y = 0.16
        marker.scale.z = 0.16
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = min(float(magnitude) / 10000.0, 1.0)
        marker.color.b = 0.0
        return marker


def main():
    HydrophoneTDOAToAngle()
    rospy.spin()


if __name__ == "__main__":
    main()

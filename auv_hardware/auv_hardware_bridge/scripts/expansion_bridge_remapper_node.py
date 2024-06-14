#!/usr/bin/env python3
import rospy
import geometry_msgs.msg
import std_msgs.msg


class ExpansionBridgeRemapperNode:
    def __init__(self):
        self.pressure_sensor_position_covariance = rospy.get_param(
            "~pressure_sensor_position_covariance"
        )

        self.pose_pub = rospy.Publisher(
            "pose", geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1
        )
        rospy.Subscriber("depth", std_msgs.msg.Float32, self.depth_callback)

    def depth_callback(self, msg: std_msgs.msg.Float32) -> None:
        pose = geometry_msgs.msg.PoseWithCovarianceStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "odom"
        pose.pose.pose.position.z = msg.data
        pose.pose.covariance[14] = self.pressure_sensor_position_covariance
        self.pose_pub.publish(pose)


if __name__ == "__main__":
    rospy.init_node("expansion_bridge_remapper_node")
    node = ExpansionBridgeRemapperNode()
    rospy.spin()

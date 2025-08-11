#!/usr/bin/env python3
import rospy
import geometry_msgs.msg
import std_msgs.msg
from auv_msgs.msg import Imu as AuvImu
from sensor_msgs.msg import Imu, MagneticField


class ExpansionBridgeRemapperNode:
    def __init__(self):
        self.pressure_sensor_position_covariance = rospy.get_param(
            "~pressure_sensor_position_covariance"
        )

        self.pose_pub = rospy.Publisher(
            "pose", geometry_msgs.msg.PoseWithCovarianceStamped, queue_size=1
        )
        self.imu_pub = rospy.Publisher("bno/data", Imu, queue_size=1)
        self.mag_pub = rospy.Publisher("bno/mag", MagneticField, queue_size=1)

        rospy.Subscriber("depth", std_msgs.msg.Float32, self.depth_callback)
        rospy.Subscriber("bno_raw", AuvImu, self.imu_callback)

    def imu_callback(self, msg: AuvImu) -> None:
        now = rospy.Time.now()

        imu_msg = Imu()
        imu_msg.header.stamp = now
        imu_msg.header.frame_id = "taluy_mini/base_link/imu_link"
        imu_msg.linear_acceleration.x = msg.linear_acceleration[0]
        imu_msg.linear_acceleration.y = msg.linear_acceleration[1]
        imu_msg.linear_acceleration.z = msg.linear_acceleration[2]
        imu_msg.angular_velocity.x = msg.angular_velocity[0]
        imu_msg.angular_velocity.y = msg.angular_velocity[1]
        imu_msg.angular_velocity.z = msg.angular_velocity[2]
        self.imu_pub.publish(imu_msg)

        mag_msg = MagneticField()
        mag_msg.header.stamp = now
        mag_msg.header.frame_id = "taluy_mini/base_link/imu_link"
        mag_msg.magnetic_field.x = msg.magnetic_field[0]
        mag_msg.magnetic_field.y = msg.magnetic_field[1]
        mag_msg.magnetic_field.z = msg.magnetic_field[2]
        self.mag_pub.publish(mag_msg)

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

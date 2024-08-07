
#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import math
from std_srvs.srv import Trigger, TriggerResponse
from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler
import angles


class RotatingTransformBroadcaster:
    def __init__(self):
        self.static_frame = 'red_buoy_link'
        self.dynamic_frame = rospy.get_param(
            '~dynamic_frame', 'rotating_frame')
        self.target_frame = 'taluy/base_link'

        self.rotation_speed = 0.1  # radians per second
        self.radius = rospy.get_param('~radius', 1.25)  # meters

        self.br = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.timer = None
        self.angle = 0.0
        self.start_angle = 0.0

        self.service = rospy.Service(
            'start_rotation', Trigger, self.handle_trigger)

    def handle_trigger(self, req):
        if self.timer is None:
            if self.lookup_initial_position():
                self.timer = rospy.Timer(
                    rospy.Duration(0.1), self.broadcast_transform)
                rospy.loginfo("Started broadcasting transform.")
                return TriggerResponse(success=True, message="Started rotation.")
            else:
                return TriggerResponse(success=False, message="Failed to lookup initial position.")
        else:
            rospy.logwarn("Rotation already started.")
            return TriggerResponse(success=False, message="Rotation already started.")

    def lookup_initial_position(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.static_frame, self.target_frame, rospy.Time(0), rospy.Duration(1.0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            # Find the nearest point on the circle to (x, y)
            distance = math.sqrt(x**2 + y**2)
            if distance != 0:
                self.angle = math.atan2(y, x)
            else:
                self.angle = 0.0
            self.start_angle = angles.normalize_angle(self.angle)

            rospy.loginfo(
                f"Initial position: x={x}, y={y}, angle={self.angle}")
            return True
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Failed to lookup transform.")
            return False

    def broadcast_transform(self, event):
        self.angle += self.rotation_speed * 0.1

        # Calculate the relative angle traversed since the start
        total_travelled_angle = (self.angle - self.start_angle)
        if total_travelled_angle >= 2 * math.pi:
            # Stop the timer and reset the state when a full rotation is complete
            self.timer.shutdown()
            self.timer = None
            self.angle = 0.0
            rospy.loginfo(
                "Completed a full rotation and stopped broadcasting.")
            return

        x = self.radius * math.cos(angles.normalize_angle(self.angle))
        y = self.radius * math.sin(angles.normalize_angle(self.angle))
        yaw = angles.normalize_angle(self.angle) + math.pi

        quat = quaternion_from_euler(0, 0, yaw)

        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.static_frame
        transform.child_frame_id = self.dynamic_frame
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        self.br.sendTransform(transform)


if __name__ == '__main__':
    rospy.init_node('rotating_transform_broadcaster')
    broadcaster = RotatingTransformBroadcaster()
    rospy.spin()

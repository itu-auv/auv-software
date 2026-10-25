#!/usr/bin/env python3

import rospy
from std_msgs.msg import UInt16


class GripperService:
    def __init__(self):
        rospy.init_node("gripper_service")
        self.angle_sub = rospy.Subscriber(
            "actuators/gripper1/set_angle", UInt16, self.handle_set_angle
        )
        self.gripper_angle_sub = rospy.Subscriber(
            "actuators/gripper/set_angle", UInt16, self.handle_set_angle
        )

    def handle_set_angle(self, msg):
        rospy.loginfo("[gripper_service] Received angle command: %d", msg.data)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        GripperService().run()
    except rospy.ROSInterruptException:
        pass

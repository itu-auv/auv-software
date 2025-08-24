#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import UInt8
from auv_msgs.srv import SendAcoustic, SendAcousticResponse


class AcousticTransmitterNode:
    def __init__(self):
        rospy.init_node("acoustic_transmitter_node")

        self.data_to_transmit = None
        self.is_transmitting = False

        self.acoustic_publisher = rospy.Publisher(
            "/taluy/modem/data/tx", UInt8, queue_size=10
        )

        self.trigger_service = rospy.Service(
            "send_acoustic_data", SendAcoustic, self.handle_send_request
        )

    def handle_send_request(self, request):
        if 1 <= request.data <= 8:
            self.data_to_transmit = request.data
            self.is_transmitting = True
            message = f"Acoustic transmission of {self.data_to_transmit} started."
            rospy.loginfo(message)
            return SendAcousticResponse(success=True, message=message)
        else:
            message = f"Invalid data: {request.data}. Must be between 1 and 8."
            rospy.logwarn(message)
            return SendAcousticResponse(success=False, message=message)

    def spin(self):
        rate = rospy.Rate(0.2)

        while not rospy.is_shutdown():
            if self.is_transmitting and self.data_to_transmit is not None:
                msg = UInt8()
                msg.data = self.data_to_transmit
                try:
                    self.acoustic_publisher.publish(msg)
                    rospy.loginfo(f"Published acoustic data: {self.data_to_transmit}")
                except Exception as e:
                    rospy.logerr(f"Error publishing acoustic data: {e}")
            rate.sleep()


if __name__ == "__main__":
    try:
        node = AcousticTransmitterNode()
        node.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("AcousticTransmitterNode interrupted")
        pass

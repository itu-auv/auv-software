#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import rospy
import threading
from std_msgs.msg import UInt8
from std_srvs.srv import Trigger, TriggerResponse


class AcousticTransmitterNode:
    def __init__(self):
        self.is_enabled = False
        self.transmission_rate = 1.0  # Hz

        rospy.init_node("acoustic_transmitter_node")

        # Publisher for acoustic modem (same topic as in acoustic.py)
        self.acoustic_publisher = rospy.Publisher(
            "/taluy/modem/data/tx", UInt8, queue_size=10
        )

        # Service to start/trigger acoustic transmission
        self.trigger_service = rospy.Service(
            "start_acoustic_transmission", Trigger, self.handle_trigger_service
        )

        # Parameters
        self.min_value = rospy.get_param("~min_value", 1)
        self.max_value = rospy.get_param("~max_value", 8)

    def handle_trigger_service(self, request: Trigger) -> TriggerResponse:
        """Handle the trigger service to start/stop acoustic transmission."""
        self.is_enabled = True

        message = f"Acoustic transmission started."
        rospy.loginfo(message)

        return TriggerResponse(success=True, message=message)

    def spin(self):
        """Main loop for acoustic transmission."""
        rate = rospy.Rate(0.1)

        while not rospy.is_shutdown():
            if not self.is_enabled:
                continue

            # Generate random data
            acoustic_data = random.randint(self.min_value, self.max_value)

            # Create and publish message
            msg = UInt8()
            msg.data = acoustic_data

            try:
                self.acoustic_publisher.publish(msg)
                rospy.loginfo(f"Published acoustic data: {acoustic_data}")
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

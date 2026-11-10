#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from auv_common_lib import ros
from rclpy.node import Node
from std_msgs.msg import UInt8
from auv_msgs.srv import SendAcoustic


class AcousticTransmitterNode(Node):
    def __init__(self):
        super().__init__("acoustic_transmitter_node")

        self.data_to_transmit = None
        self.is_transmitting = False

        self.acoustic_publisher = self.create_publisher(UInt8, "modem/data/tx", 10)

        self.trigger_service = self.create_service(
            SendAcoustic, "send_acoustic_data", self.handle_send_request
        )

    def handle_send_request(self, request, response):
        if 1 <= request.data <= 8:
            self.data_to_transmit = request.data
            self.is_transmitting = True
            message = f"Acoustic transmission of {self.data_to_transmit} started."
            self.get_logger().info(message)
            return SendAcoustic.Response(success=True, message=message)
        else:
            message = f"Invalid data: {request.data}. Must be between 1 and 8."
            self.get_logger().warning(message)
            return SendAcoustic.Response(success=False, message=message)

    def spin(self):
        rate = self.create_rate(0.2)

        while rclpy.ok():
            if self.is_transmitting and self.data_to_transmit is not None:
                msg = UInt8()
                msg.data = self.data_to_transmit
                try:
                    self.acoustic_publisher.publish(msg)
                    self.get_logger().info(
                        f"Published acoustic data: {self.data_to_transmit}"
                    )
                except Exception as e:
                    self.get_logger().error(f"Error publishing acoustic data: {e}")
            rate.sleep()


def main():
    ros.init()
    node = AcousticTransmitterNode()
    ros.spin(node, node.spin)


if __name__ == "__main__":
    main()

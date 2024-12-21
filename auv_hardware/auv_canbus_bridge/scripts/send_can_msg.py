#!/usr/bin/env python3

import rospy
import socket
import struct


def send_can_message():
    # Get the CAN interface name from parameter server
    can_interface = rospy.get_param("~can_interface", "vcan0")  # Default to "vcan0"

    # Set up CAN socket
    can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)

    try:
        # Bind to the interface
        can_socket.bind((can_interface,))
    except OSError:
        rospy.logerr(f"Failed to bind to {can_interface}")
        return

    # Extended CAN ID and payload
    can_id = 0x00112233 | socket.CAN_EFF_FLAG
    data = [0xFF, 0xFA, 0xFF, 0xFA, 0x00, 0x10, 0x00, 0x10]

    # Prepare CAN frame (ID + DLC + data)
    can_frame = struct.pack("=IB3x8s", can_id, len(data), bytes(data))

    # Send CAN frame
    can_socket.send(can_frame)
    rospy.loginfo(f"CAN message sent on interface: {can_interface}")


if __name__ == "__main__":
    rospy.init_node("send_can_msg")
    send_can_message()

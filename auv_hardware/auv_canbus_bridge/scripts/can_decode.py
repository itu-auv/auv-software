#!/usr/bin/env python3
import rospy
import socket
import struct


def decode_can_payload(data):
    # Decode 8 bytes of CAN data into four 16-bit integers
    values = [
        int.from_bytes(data[i : i + 2], byteorder="big") for i in range(0, len(data), 2)
    ]
    return values


def listen_can():
    rospy.init_node("can_decoder", anonymous=True)

    # Set up the CAN socket
    can_socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    can_socket.bind(("vcan0",))

    rospy.loginfo("Listening for CAN messages on vcan0...")
    while not rospy.is_shutdown():
        frame, _ = can_socket.recvfrom(16)
        can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
        decoded_values = decode_can_payload(data[:can_dlc])
        rospy.loginfo(f"CAN ID: {hex(can_id)} Decoded Values: {decoded_values}")


if __name__ == "__main__":
    try:
        listen_can()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
import rospy
import socket
import struct


def float16_to_float32_debug(f16):
    # Extract components
    sign = (f16 & 0x8000) >> 15
    exponent = (f16 & 0x7C00) >> 10
    mantissa = f16 & 0x03FF

    if exponent == 0:
        if mantissa == 0:
            # Zero
            return (-1) ** sign * 0.0
        else:
            # Subnormal number
            value = (-1) ** sign * (mantissa / 1024.0) * 2 ** (-14)
            return value
    elif exponent == 0x1F:
        if mantissa == 0:
            # Infinity
            value = float("inf") if sign == 0 else float("-inf")
            return value
        else:
            # NaN
            return float("nan")
    else:
        # Normalized number
        value = (-1) ** sign * (1 + mantissa / 1024.0) * 2 ** (exponent - 15)
        return value


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
    can_socket.bind(("can1",))

    rospy.loginfo("Listening for CAN messages on vcan0...")
    while not rospy.is_shutdown():
        frame, _ = can_socket.recvfrom(16)
        can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
        decoded_values = decode_can_payload(data[:can_dlc])
        rospy.loginfo(
            f"CAN ID: {hex(can_id)} Decoded Values: {[float16_to_float32_debug(v) for v in decoded_values]}"
        )


if __name__ == "__main__":
    try:
        listen_can()
    except rospy.ROSInterruptException:
        pass

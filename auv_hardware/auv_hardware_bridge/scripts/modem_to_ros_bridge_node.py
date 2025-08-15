#!/usr/bin/env python3
import rospy
import serial
from std_msgs.msg import UInt8

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200


def tx_callback(msg):
    """Send :AUV<raw_byte> to serial when modem/tx is published."""
    send_bytes = b":AUV" + bytes([msg.data])  # raw uint8 byte
    ser.write(send_bytes)
    rospy.loginfo(f"TX: {list(send_bytes)}")  # shows bytes in decimal


if __name__ == "__main__":
    rospy.init_node("simple_serial_bridge")

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    rospy.loginfo(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

    rx_pub = rospy.Publisher("modem/rx", UInt8, queue_size=10)
    rospy.Subscriber("modem/tx", UInt8, tx_callback)

    rate = rospy.Rate(10)  # 10 Hz
    try:
        while not rospy.is_shutdown():
            if ser.in_waiting > 0:
                data = ser.read_until(
                    expected=b"\n"
                )  # read until newline or packet end
                if data.startswith(b":AUV") and len(data) >= 5:
                    value = data[4]  # 5th byte is the raw uint8
                    rx_pub.publish(value)
                    rospy.loginfo(f"RX: {value}")
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        ser.close()

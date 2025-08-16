#!/usr/bin/env python3
import rospy
import serial
from std_msgs.msg import UInt8

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200


if __name__ == "__main__":
    rospy.init_node("simple_serial_bridge")

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    rospy.loginfo(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

    rx_pub = rospy.Publisher("modem/rx", UInt8, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    try:
        while not rospy.is_shutdown():
            if ser.in_waiting > 0:
                data = ser.read(5)
                rospy.logerr(f"recv {data}")
                if data.startswith(b":AUV") and len(data) >= 5:
                    value = data[4]  # 5th byte is the raw uint8
                    rospy.logerr(f"Received data: {data}")
                    rx_pub.publish(value)
                    rospy.loginfo(f"RX: {value}")
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.logerr("birseyler")
        pass
    finally:
        ser.close()

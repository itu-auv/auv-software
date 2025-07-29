#!/usr/bin/env python3

import rospy
import serial
from std_msgs.msg import UInt8

HEADER = b':AUV'  

class SerialBridge:
    def __init__(self):
        rospy.init_node('modem_to_ros_bridge')

        self.port = rospy.get_param("~serial_port", "/dev/auv_taluymini_modem")
        self.baudrate = rospy.get_param("~baudrate", 921600)

        # TX packet: 5 bytes total
        self.tx_packet = bytearray(b':AUV\x00')

        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            rospy.loginfo(f"Opened serial port: {self.port} at {self.baudrate} baud")
        except serial.SerialException as e:
            rospy.logerr(f"Serial port error: {e}")
            raise

        # Publishers & Subscribers
        self.rx_pub = rospy.Publisher('modem/data/rx', UInt8, queue_size=10)
        rospy.Subscriber('modem/data/tx', UInt8, self.tx_callback)

    def tx_callback(self, msg: UInt8):
        self.tx_packet[-1] = msg.data
        try:
            self.ser.write(self.tx_packet)
            rospy.loginfo(f"Sent TX packet: {list(self.tx_packet)}")
        except Exception as e:
            rospy.logerr(f"Serial write failed: {e}")

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.ser.in_waiting >= 5: # bu böyle mi
                raw = self.ser.read(5)
                if raw.startswith(HEADER) and len(raw) == 5:
                    payload = raw[-1]
                    self.rx_pub.publish(UInt8(data=payload))
                    rospy.loginfo(f"Received RX byte: {payload}")
            rate.sleep()

        self.ser.close()

if __name__ == '__main__':
    try:
        bridge = SerialBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        pass

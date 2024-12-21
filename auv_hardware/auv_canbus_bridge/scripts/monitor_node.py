#!/usr/bin/env python3

import rospy
import auv_msgs.msg
import socket
import struct
import threading
from auv_canbus_msgs.msg import QuadDrivePulse


class CANIdentifier:
    CAN_EFF_FLAG = 0x80000000

    def __init__(self, priority: int, node_id: int, function: int, endpoint: int):
        self.priority = priority
        self.node_id = node_id
        self.function = function
        self.endpoint = endpoint

    def from_bytes(can_id: int):
        can_id &= ~CANIdentifier.CAN_EFF_FLAG

        priority = (can_id >> 24) & 0xFF
        node_id = (can_id >> 17) & 0x7F
        function = (can_id >> 16) & 0x01
        endpoint = can_id & 0xFFFF

        return CANIdentifier(priority, node_id, function, endpoint)

    def node_id_name(self):
        lookup = {
            auv_msgs.msg.CanFrame.NODE_ID_EXPANSION_BOARD: "EXPANSION",
            auv_msgs.msg.CanFrame.NODE_ID_MAINBOARD: "MAINBOARD",
            auv_msgs.msg.CanFrame.NODE_ID_PROPULSION_BOARD: "PROPULSION",
            auv_msgs.msg.CanFrame.NODE_ID_ROS_BRIDGE: "ROSBRIDGE",
        }

        return lookup.get(self.node_id, "UNKNOWN")

    def to_bytes(self) -> int:
        identifier = 0

        identifier |= (self.priority & 0xFF) << 24
        identifier |= (self.node_id & 0x7F) << 17
        identifier |= (self.function & 0x01) << 16
        identifier |= self.endpoint & 0xFFFF

        identifier |= CANIdentifier.CAN_EFF_FLAG

        return identifier

    def __str__(self):
        # example return
        # TO:{node_id}F:{W/R}E:{endpoint}
        function = "W" if self.function == 0 else "R"
        return f"DST={self.node_id_name()} {function}"


class CANMessage:
    ENDPOINT_ID = None

    def from_bytes(data: bytes) -> "CANMessage":
        raise NotImplementedError

    def __str__(self):
        return "CAN Message"


class FirstQuadThrusterCommand(CANMessage):
    ENDPOINT_ID = auv_msgs.msg.CanFrame.ENDPOINT_FIRST_QUAD_THRUSTER_COMMAND

    def __init__(self, channels: list = [0, 0, 0, 0]):
        self.channels = channels

    def from_bytes(data: bytes) -> "FirstQuadThrusterCommand":
        return FirstQuadThrusterCommand(
            [
                int.from_bytes(data[i : i + 2], byteorder="big")
                for i in range(0, len(data), 2)
            ]
        )

    def __str__(self):
        return f"[Thruster Command (0-4)] {self.channels}"


class CANMessageFactory:
    @staticmethod
    def from_bytes(can_id: CANIdentifier, data: bytes) -> CANMessage:
        if can_id.endpoint == FirstQuadThrusterCommand.ENDPOINT_ID:
            return FirstQuadThrusterCommand.from_bytes(data)

        return CANMessage()


class CANMessageLogger:

    def on_receive(self, can_id: int, can_dlc: int, data: bytes):
        identifier = CANIdentifier.from_bytes(can_id)
        can_message = CANMessageFactory.from_bytes(identifier, data)

        rospy.loginfo(f"[{identifier}] {can_message}")


class CANMonitoringROS:
    CAN_EFF_FLAG = 0x80000000

    def __init__(self):
        self.interface = rospy.get_param("~interface", "vcan0")
        self.socket_lock = threading.Lock()
        self.socket = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        self.socket.bind((self.interface,))
        self.socket.setblocking(False)
        self.can_message_publisher = rospy.Publisher(
            "canbus/incoming_frame", auv_msgs.msg.CanFrame, queue_size=10
        )
        rospy.Subscriber(
            "canbus/outgoing_frame",
            auv_msgs.msg.CanFrame,
            self.outgoing_frame_callback,
        )
        self.timer = rospy.Timer(rospy.Duration(0.01), self.handle_socket_receive)

    def outgoing_frame_callback(self, can_frame: auv_msgs.msg.CanFrame):

        can_id = CANIdentifier(
            can_frame.priority,
            can_frame.node_id,
            can_frame.function,
            can_frame.endpoint,
        ).to_bytes()

        can_dlc = can_frame.data_length
        data = can_frame.data.ljust(8, b"\x00")

        frame = struct.pack("=IB3x8s", can_id, can_dlc, data)

        with self.socket_lock:
            print(f"Sending frame: {frame}")
            self.socket.send(frame)

    def can_frame_socket_callback(self, can_id: int, can_dlc: int, data: bytes):
        identifier = CANIdentifier.from_bytes(can_id)

        can_frame = auv_msgs.msg.CanFrame()
        can_frame.priority = identifier.priority
        can_frame.node_id = identifier.node_id
        can_frame.function = identifier.function
        can_frame.endpoint = identifier.endpoint
        can_frame.data = data
        can_frame.data_length = can_dlc

        CANMessageLogger().on_receive(can_id, can_dlc, data)

        self.can_message_publisher.publish(can_frame)

    def handle_socket_receive(self, event: None):
        while not rospy.is_shutdown():
            with self.socket_lock:
                try:
                    frame = self.socket.recv(16)
                except socket.timeout:
                    continue
                except BlockingIOError:
                    continue

            can_id, can_dlc, data = struct.unpack("=IB3x8s", frame)
            self.can_frame_socket_callback(can_id, can_dlc, data[:can_dlc])

    def close(self):
        self.socket.close()


if __name__ == "__main__":
    rospy.init_node("can_monitor_node")
    node = CANMonitoringROS()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.close()

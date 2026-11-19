#!/usr/bin/env python3

import rclpy
from auv_common_lib.ros import latched_qos, get_node, ROSException, Rate
from rclpy.duration import Duration
from rclpy.exceptions import ROSInterruptException
import smach
from std_msgs.msg import UInt8MultiArray


DEFAULT_ACOUSTIC_RX_TOPIC = "acoustic/modem/received"
DEFAULT_ACOUSTIC_TX_TOPIC = "acoustic/modem/transmit"


def _normalize_data(data):
    if data is None:
        return None
    if isinstance(data, UInt8MultiArray):
        return list(data.data)
    if isinstance(data, (list, tuple)):
        return [int(value) for value in data]
    return [int(data)]


class AcousticTransmitter(smach.State):
    def __init__(self, acoustic_data=None, topic_name=DEFAULT_ACOUSTIC_TX_TOPIC):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.acoustic_data = _normalize_data(acoustic_data)
        self.topic_name = topic_name
        self.acoustic_pub = get_node().create_publisher(
            UInt8MultiArray, self.topic_name, latched_qos(10)
        )

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        if self.acoustic_data is None:
            get_node().get_logger().error(
                "AcousticTransmitter has no acoustic_data to publish"
            )
            return "aborted"

        msg = UInt8MultiArray(data=self.acoustic_data)

        try:
            for i in range(0, 5):
                self.acoustic_pub.publish(msg)
                get_node().get_clock().sleep_for(Duration(seconds=1))
        except ROSException as e:
            get_node().get_logger().error(f"Error publishing acoustic data: {e}")
            return "aborted"

        get_node().get_logger().info(
            f"Published acoustic data once on {self.topic_name}: {self.acoustic_data}"
        )
        return "succeeded"


class AcousticReceiver(smach.State):

    def __init__(
        self,
        expected_data=None,
        timeout=None,
        topic_name=DEFAULT_ACOUSTIC_RX_TOPIC,
        accept_any_data=False,
    ):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.expected_data = _normalize_data(expected_data)
        self.timeout = timeout
        self.topic_name = topic_name
        self.accept_any_data = accept_any_data
        self.data_received = False
        self.acoustic_sub = None

        get_node().get_logger().info(
            f"AcousticReceiver initialized - topic: {self.topic_name}, expected: {self.expected_data}, accept_any_data: {self.accept_any_data}, timeout: {timeout}s"
        )

    def acoustic_callback(self, msg):
        received_data = list(msg.data)
        get_node().get_logger().debug(f"Received acoustic data: {received_data}")

        if self.accept_any_data or self.expected_data is None:
            self.data_received = True
            get_node().get_logger().info(f"Acoustic data received: {received_data}")
        elif received_data == self.expected_data:
            self.data_received = True
            get_node().get_logger().info(
                f"Expected acoustic data received: {received_data}"
            )

    def execute(self, userdata):
        get_node().get_logger().info(
            f"Starting acoustic reception on {self.topic_name} - waiting for: {self.expected_data}, accept_any_data: {self.accept_any_data}"
        )

        self.data_received = False
        self.acoustic_sub = get_node().create_subscription(
            UInt8MultiArray, self.topic_name, self.acoustic_callback, 10
        )

        if self.timeout is not None:
            start_time = get_node().get_clock().now()
            timeout_time = start_time + Duration(seconds=self.timeout)
        else:
            timeout_time = None

        rate = Rate(get_node(), 10)

        try:
            while not self.data_received:
                if self.preempt_requested():
                    self.service_preempt()
                    get_node().get_logger().info("Acoustic reception preempted")
                    return "preempted"

                if (
                    timeout_time is not None
                    and get_node().get_clock().now() >= timeout_time
                ):
                    get_node().get_logger().info(
                        f"Acoustic reception timeout after {self.timeout}s - continuing mission"
                    )
                    return "succeeded"

                rate.sleep()

        except ROSInterruptException:
            get_node().get_logger().info("Acoustic reception interrupted")
            return "aborted"
        except Exception as e:
            get_node().get_logger().error(f"Error during acoustic reception: {e}")
            return "aborted"
        finally:
            if self.acoustic_sub is not None:
                get_node().destroy_subscription(self.acoustic_sub)
                self.acoustic_sub = None

        get_node().get_logger().info(
            "Acoustic reception completed - expected data received"
        )
        return "succeeded"

#!/usr/bin/env python3

import rospy
import smach
from std_msgs.msg import UInt8MultiArray


DEFAULT_ACOUSTIC_RX_TOPIC = "acoustic/modem/received"
DEFAULT_ACOUSTIC_TX_TOPIC = "acoustic/modem/transmitted"


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
        self.acoustic_pub = rospy.Publisher(
            self.topic_name, UInt8MultiArray, queue_size=10, latch=True
        )

    def execute(self, userdata):
        if self.preempt_requested():
            self.service_preempt()
            return "preempted"

        if self.acoustic_data is None:
            rospy.logerr("AcousticTransmitter has no acoustic_data to publish")
            return "aborted"

        msg = UInt8MultiArray(data=self.acoustic_data)

        try:
            self.acoustic_pub.publish(msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Error publishing acoustic data: {e}")
            return "aborted"

        rospy.loginfo(
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

        rospy.loginfo(
            f"AcousticReceiver initialized - topic: {self.topic_name}, expected: {self.expected_data}, accept_any_data: {self.accept_any_data}, timeout: {timeout}s"
        )

    def acoustic_callback(self, msg):
        received_data = list(msg.data)
        rospy.logdebug(f"Received acoustic data: {received_data}")

        if self.accept_any_data or self.expected_data is None:
            self.data_received = True
            rospy.loginfo(f"Acoustic data received: {received_data}")
        elif received_data == self.expected_data:
            self.data_received = True
            rospy.loginfo(f"Expected acoustic data received: {received_data}")

    def execute(self, userdata):
        rospy.loginfo(
            f"Starting acoustic reception on {self.topic_name} - waiting for: {self.expected_data}, accept_any_data: {self.accept_any_data}"
        )

        self.data_received = False
        self.acoustic_sub = rospy.Subscriber(
            self.topic_name, UInt8MultiArray, self.acoustic_callback
        )

        if self.timeout is not None:
            start_time = rospy.Time.now()
            timeout_time = start_time + rospy.Duration(self.timeout)
        else:
            timeout_time = None

        rate = rospy.Rate(10)

        try:
            while not self.data_received:
                if self.preempt_requested():
                    self.service_preempt()
                    rospy.loginfo("Acoustic reception preempted")
                    return "preempted"

                if timeout_time is not None and rospy.Time.now() >= timeout_time:
                    rospy.loginfo(
                        f"Acoustic reception timeout after {self.timeout}s - continuing mission"
                    )
                    return "succeeded"

                rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("Acoustic reception interrupted")
            return "aborted"
        except Exception as e:
            rospy.logerr(f"Error during acoustic reception: {e}")
            return "aborted"
        finally:
            if self.acoustic_sub is not None:
                self.acoustic_sub.unregister()
                self.acoustic_sub = None

        rospy.loginfo("Acoustic reception completed - expected data received")
        return "succeeded"

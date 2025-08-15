#!/usr/bin/env python3

import rospy
import smach
from std_msgs.msg import UInt8


class AcousticTransmitter(smach.State):

    def __init__(self, data_value):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.data_value = data_value

        # Create publisher for acoustic modem
        self.acoustic_pub = rospy.Publisher("modem/tx", UInt8, queue_size=10)

        rospy.loginfo(f"AcousticTransmitter initialized - data: {data_value}")

    def execute(self, userdata):
        rospy.loginfo(f"Publishing acoustic data: {self.data_value}")

        # Create and publish message
        msg = UInt8()
        msg.data = self.data_value

        try:
            self.acoustic_pub.publish(msg)
            rospy.loginfo(f"Acoustic transmission completed - data: {self.data_value}")
            return "succeeded"

        except Exception as e:
            rospy.logerr(f"Error during acoustic transmission: {e}")
            return "aborted"


class AcousticReceiver(smach.State):

    def __init__(self, expected_data=None, timeout=None):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.expected_data = expected_data
        self.timeout = timeout
        self.data_received = False

        # Create subscriber for acoustic modem
        self.acoustic_sub = rospy.Subscriber("modem/rx", UInt8, self.acoustic_callback)

        rospy.loginfo(
            f"AcousticReceiver initialized - expected: {expected_data}, timeout: {timeout}s"
        )

    def acoustic_callback(self, msg):
        rospy.logdebug(f"Received acoustic data: {msg.data}")

        # Check if this is the expected data
        if self.expected_data is None:
            # Accept any data
            self.data_received = True
            rospy.loginfo(f"Acoustic data received: {msg.data}")
        elif isinstance(self.expected_data, list):
            # Check if data is in the expected list
            if msg.data in self.expected_data:
                self.data_received = True
                rospy.loginfo(f"Expected acoustic data received: {msg.data}")
        else:
            # Check if data matches expected value
            if msg.data == self.expected_data:
                self.data_received = True
                rospy.loginfo(f"Expected acoustic data received: {msg.data}")

    def execute(self, userdata):
        rospy.loginfo(
            f"Starting acoustic reception - waiting for: {self.expected_data}"
        )

        # Reset state
        self.data_received = False

        # Calculate timeout
        if self.timeout is not None:
            start_time = rospy.Time.now()
            timeout_time = start_time + rospy.Duration(self.timeout)
        else:
            timeout_time = None

        rate = rospy.Rate(10)

        try:
            while not self.data_received:
                # Check for preemption
                if self.preempt_requested():
                    self.service_preempt()
                    rospy.loginfo("Acoustic reception preempted")
                    return "preempted"

                # Check for timeout
                if timeout_time is not None and rospy.Time.now() >= timeout_time:
                    rospy.loginfo(
                        f"Acoustic reception timeout after {self.timeout}s - continuing mission"
                    )
                    return "succeeded"

                # Sleep and continue waiting
                rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("Acoustic reception interrupted")
            return "aborted"
        except Exception as e:
            rospy.logerr(f"Error during acoustic reception: {e}")
            return "aborted"

        rospy.loginfo("Acoustic reception completed - expected data received")
        return "succeeded"

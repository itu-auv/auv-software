#!/usr/bin/env python3

import rospy
import smach
from std_msgs.msg import UInt8


class AcousticState(smach.State):
    """
    State for sending acoustic modem data at specified rate and duration
    """

    def __init__(self, data_value, publish_rate, duration):
        """
        Initialize the acoustic state

        Args:
            data_value (int): The data value to publish (0-255)
            publish_rate (float): Rate in Hz at which to publish the message
            duration (float): Duration in seconds for how long to publish
        """
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])

        self.data_value = data_value
        self.publish_rate = publish_rate
        self.duration = duration

        # Create publisher for acoustic modem
        self.acoustic_pub = rospy.Publisher(
            "/taluy/modem/data/tx", UInt8, queue_size=10
        )

        rospy.loginfo(
            f"AcousticState initialized - data: {data_value}, rate: {publish_rate} Hz, duration: {duration}s"
        )

    def execute(self, userdata):
        """
        Execute the acoustic state - publish data at specified rate for given duration
        """
        rospy.loginfo(f"Starting acoustic transmission - data: {self.data_value}")

        # Create rate object
        rate = rospy.Rate(self.publish_rate)

        # Calculate end time
        start_time = rospy.Time.now()
        end_time = start_time + rospy.Duration(self.duration)

        # Create message
        msg = UInt8()
        msg.data = self.data_value

        try:
            while rospy.Time.now() < end_time:
                # Check for preemption
                if self.preempt_requested():
                    self.service_preempt()
                    rospy.loginfo("Acoustic transmission preempted")
                    return "preempted"

                # Publish the message
                self.acoustic_pub.publish(msg)
                rospy.logdebug(f"Published acoustic data: {self.data_value}")

                # Sleep according to rate
                rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("Acoustic transmission interrupted")
            return "aborted"
        except Exception as e:
            rospy.logerr(f"Error during acoustic transmission: {e}")
            return "aborted"

        rospy.loginfo(
            f"Acoustic transmission completed - published {self.data_value} for {self.duration}s at {self.publish_rate} Hz"
        )
        return "succeeded"

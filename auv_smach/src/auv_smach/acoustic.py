#!/usr/bin/env python3
"""
SMACH State wrappers for Acoustic Behavior Trees.

This module provides SMACH-compatible wrappers that internally use
py_trees behaviors for acoustic modem communication.
"""

import rospy
import smach
import py_trees
import py_trees_ros
from auv_smach.behaviors.actions import (
    AcousticTransmitBehavior,
    AcousticReceiveBehavior,
)


class AcousticTransmitter(smach.State):
    """
    SMACH State wrapper for AcousticTransmitBehavior.

    Transmits acoustic data via the acoustic modem using the underlying
    py_trees behavior.
    """

    def __init__(self, acoustic_data: int):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.acoustic_data = acoustic_data
        self.behaviour_tree = None

    def execute(self, userdata):
        rospy.loginfo(f"[AcousticTransmitter] Transmitting data: {self.acoustic_data}")

        try:
            # Create the behavior
            root = AcousticTransmitBehavior(
                name="TransmitAcoustic",
                acoustic_data=self.acoustic_data,
            )

            # Initialize the tree
            self.behaviour_tree = py_trees_ros.trees.BehaviourTree(root)
            self.behaviour_tree.setup(timeout=15.0)

            # Single tick for service call
            self.behaviour_tree.tick()
            status = root.status

            if status == py_trees.common.Status.SUCCESS:
                rospy.loginfo("[AcousticTransmitter] Transmission succeeded!")
                return "succeeded"

            if status == py_trees.common.Status.FAILURE:
                rospy.logerr("[AcousticTransmitter] Transmission failed!")
                return "aborted"

            return "aborted"

        except rospy.ROSInterruptException:
            rospy.loginfo("[AcousticTransmitter] Transmission interrupted")
            return "aborted"
        except Exception as e:
            rospy.logerr(f"[AcousticTransmitter] Error during transmission: {e}")
            return "aborted"


class AcousticReceiver(smach.State):
    """
    SMACH State wrapper for AcousticReceiveBehavior.

    Waits for acoustic data from the modem using the underlying
    py_trees behavior.
    """

    def __init__(self, expected_data=None, timeout=None):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        self.expected_data = expected_data
        self.timeout = timeout
        self.behaviour_tree = None

    def execute(self, userdata):
        rospy.loginfo(
            f"[AcousticReceiver] Waiting for data: {self.expected_data}, "
            f"timeout: {self.timeout}s"
        )

        try:
            # Create the behavior
            root = AcousticReceiveBehavior(
                name="ReceiveAcoustic",
                expected_data=self.expected_data,
                timeout=self.timeout,
            )

            # Initialize the tree
            self.behaviour_tree = py_trees_ros.trees.BehaviourTree(root)
            self.behaviour_tree.setup(timeout=15.0)

            # Tick loop
            rate = rospy.Rate(10)  # 10 Hz

            while not rospy.is_shutdown():
                if self.preempt_requested():
                    self.service_preempt()
                    self.behaviour_tree.interrupt()
                    rospy.loginfo("[AcousticReceiver] Reception preempted")
                    return "preempted"

                self.behaviour_tree.tick()
                status = root.status

                if status == py_trees.common.Status.SUCCESS:
                    rospy.loginfo("[AcousticReceiver] Reception succeeded!")
                    return "succeeded"

                if status == py_trees.common.Status.FAILURE:
                    rospy.logerr("[AcousticReceiver] Reception failed!")
                    return "aborted"

                rate.sleep()

            return "aborted"

        except rospy.ROSInterruptException:
            rospy.loginfo("[AcousticReceiver] Reception interrupted")
            return "aborted"
        except Exception as e:
            rospy.logerr(f"[AcousticReceiver] Error during reception: {e}")
            return "aborted"

#!/usr/bin/env python3
"""
Test script for the WAIT_DEPTH_ALIGNMENT state.
This script creates a simple state machine to test the depth alignment functionality.
"""

import rospy
import smach
import smach_ros
from auv_smach.common import WaitDepthAlignment, SetDepthState


def main():
    rospy.init_node("test_wait_depth_alignment")

    # Create a simple state machine
    sm = smach.StateMachine(outcomes=["succeeded", "aborted", "preempted"])

    with sm:
        # First, set a target depth
        smach.StateMachine.add(
            "SET_DEPTH",
            SetDepthState(
                depth=-1.5,  # Target depth of 1 meter
                sleep_duration=2.0,  # Wait 2 seconds after setting depth
                frame_id="taluy/base_link",  # Use odom frame
            ),
            transitions={
                "succeeded": "WAIT_DEPTH_ALIGNMENT",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

        # Then wait for depth alignment
        smach.StateMachine.add(
            "WAIT_DEPTH_ALIGNMENT",
            WaitDepthAlignment(
                depth_threshold=0.05,  # 5cm threshold
                confirm_duration=2.0,  # Maintain alignment for 2 seconds
                timeout=10.0,  # 10 second timeout
                rate_hz=10,  # Check at 10Hz
            ),
            transitions={
                "succeeded": "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

    # Create and start the introspection server for visualization
    sis = smach_ros.IntrospectionServer("test_wait_depth_alignment", sm, "/SM_ROOT")
    sis.start()

    # Execute the state machine
    rospy.loginfo("Starting WAIT_DEPTH_ALIGNMENT test...")
    outcome = sm.execute()
    rospy.loginfo(f"State machine finished with outcome: {outcome}")

    # Stop the introspection server
    sis.stop()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

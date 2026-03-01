#!/usr/bin/env python3
"""
Valve Test — Standalone
-----------------------
Sadece valve task'ını çalıştıran bağımsız test script'i.
main_state_machine'den bağımsız çalışır.

Kullanım:
    rosrun auv_smach valve_test.py
"""

import rospy
import smach
from auv_smach.valve import ValveTaskState


def main():
    rospy.init_node("valve_test_state_machine")

    # Valve parametreleri
    valve_depth = rospy.get_param("~valve_depth", -1.25)
    valve_coarse_approach_frame = rospy.get_param("~valve_coarse_approach_frame", "valve_coarse_approach_frame")
    valve_approach_frame = rospy.get_param("~valve_approach_frame", "valve_approach_frame")
    valve_contact_frame = rospy.get_param("~valve_contact_frame", "valve_contact_frame")
    valve_exit_angle = rospy.get_param("~valve_exit_angle", 0.0)
    use_ground_truth = rospy.get_param("~use_ground_truth", False)

    rospy.loginfo("=== VALVE TEST (3-Phase Approach) ===")
    rospy.loginfo(f"  depth:                {valve_depth}")
    rospy.loginfo(f"  coarse_approach_frame: {valve_coarse_approach_frame}")
    rospy.loginfo(f"  approach_frame:        {valve_approach_frame}")
    rospy.loginfo(f"  contact_frame:         {valve_contact_frame}")
    rospy.loginfo(f"  exit_angle:            {valve_exit_angle}")
    rospy.loginfo(f"  use_ground_truth:      {use_ground_truth}")

    sm = smach.StateMachine(outcomes=["succeeded", "preempted", "aborted"])

    with sm:
        smach.StateMachine.add(
            "VALVE_TASK",
            ValveTaskState(
                valve_depth=valve_depth,
                valve_coarse_approach_frame=valve_coarse_approach_frame,
                valve_approach_frame=valve_approach_frame,
                valve_contact_frame=valve_contact_frame,
                valve_exit_angle=valve_exit_angle,
                use_ground_truth=use_ground_truth,
            ),
            transitions={
                "succeeded": "succeeded",
                "preempted": "preempted",
                "aborted": "aborted",
            },
        )

    outcome = sm.execute()
    rospy.loginfo(f"Valve test finished with outcome: {outcome}")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

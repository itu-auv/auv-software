#!/usr/bin/env python3
"""
Valve Task Test Script
-----------------------
Sadece valve görevini bağımsız olarak test etmek için.

Kullanım:
  rosrun auv_smach test_valve.py

Gereksinimler (çalışıyor olmalı):
  - Simülasyon (Gazebo)
  - auv_mapping/start.launch (object_map_tf_server + valve_trajectory_publisher)
  - auv_vision/valve_detector.launch (valve detection)
  - Controller (align_frame servisleri)
"""

import rospy
from auv_smach.valve import ValveTaskState


if __name__ == "__main__":
    rospy.init_node("test_valve_task")

    valve_depth = rospy.get_param("~valve_depth", -1.0)

    rospy.loginfo(f"=== VALVE TEST: depth={valve_depth}m ===")
    rospy.loginfo("Starting in 2 seconds...")
    rospy.sleep(2.0)

    state = ValveTaskState(valve_depth=valve_depth)
    outcome = state.execute(None)

    rospy.loginfo(f"=== VALVE TEST FINISHED: {outcome} ===")

#!/usr/bin/env python

import rospy
import rosnode
import os
from std_srvs.srv import SetBool, SetBoolRequest, Trigger, TriggerRequest
from auv_msgs.srv import SetDetectionFocus, SetDetectionFocusRequest


class SmachMonitor:
    def __init__(self):
        rospy.init_node("smach_monitor_node", anonymous=True)

        self.main_sm_node_name = "main_state_machine"
        self.teleop_node_name = "joystick_node"

        self.align_frame_service_name = "control/align_frame/cancel"
        self.heading_control_service_name = "set_heading_control"
        self.detection_focus_service_name = "vision/set_front_camera_focus"

        self.align_frame_service = rospy.ServiceProxy(
            self.align_frame_service_name, Trigger
        )
        self.heading_control_service = rospy.ServiceProxy(
            self.heading_control_service_name, SetBool
        )
        self.detection_focus_service = rospy.ServiceProxy(
            self.detection_focus_service_name, SetDetectionFocus
        )

        self.smach_is_active = False
        self.rate = rospy.Rate(1)  # 1hz

    def run(self):
        while not rospy.is_shutdown():
            if self.is_node_active(self.main_sm_node_name):
                if not self.smach_is_active:
                    rospy.loginfo("Main state machine has started.")
                    self.smach_is_active = True
                    if self.is_node_active(self.teleop_node_name):
                        rospy.loginfo("Teleop node is running. Killing it.")
                        os.system("rosnode kill " + self.teleop_node_name)
            else:
                if self.smach_is_active:
                    rospy.loginfo("Main state machine has died.")
                    self.smach_is_active = False
                    self.on_smach_death()

            self.rate.sleep()

    def is_node_active(self, node_name):
        try:
            nodes = rosnode.get_node_names()
        except rosnode.ROSNodeIOException:
            # ROS master is not available (shutting down)
            return False

        for node in nodes:
            if node.endswith(node_name):
                try:
                    rosnode.rosnode_ping(node, max_count=1)
                    return True
                except rosnode.ROSNodeUnreachable:
                    continue
        return False

    def on_smach_death(self):
        rospy.loginfo("Executing recovery actions.")

        # 1. Cancel align frame controller
        try:
            rospy.wait_for_service(self.align_frame_service_name, timeout=2.0)
            req = TriggerRequest()
            self.align_frame_service(req)
            rospy.loginfo("Cancelled align frame controller.")
        except (
            rospy.ServiceException,
            rospy.ROSException,
            rospy.ROSInterruptException,
        ) as e:
            rospy.logerr("Service call to cancel align frame controller failed: %s" % e)

        # 2. Enable heading control
        try:
            rospy.wait_for_service(self.heading_control_service_name, timeout=2.0)
            req = SetBoolRequest(data=True)
            self.heading_control_service(req)
            rospy.loginfo("Enabled heading control.")
        except (
            rospy.ServiceException,
            rospy.ROSException,
            rospy.ROSInterruptException,
        ) as e:
            rospy.logerr("Service call to enable heading control failed: %s" % e)

        # 3. Set DetectionFocus to all
        try:
            rospy.wait_for_service(self.detection_focus_service_name, timeout=2.0)
            req = SetDetectionFocusRequest(focus_object="none")
            self.detection_focus_service(req)
            rospy.loginfo("Set DetectionFocus to 'all'.")
        except (
            rospy.ServiceException,
            rospy.ROSException,
            rospy.ROSInterruptException,
        ) as e:
            rospy.logerr("Service call to set detection focus failed: %s" % e)


if __name__ == "__main__":
    try:
        monitor = SmachMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass

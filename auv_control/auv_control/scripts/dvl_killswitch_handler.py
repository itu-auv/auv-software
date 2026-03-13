#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, SetBoolRequest


class DvlKillswitchHandler:
    def __init__(self):
        self.previous_status = None

        rospy.wait_for_service("dvl/enable", timeout=5.0)
        self.dvl_enable_service = rospy.ServiceProxy("dvl/enable", SetBool)

        self.status_sub = rospy.Subscriber(
            "propulsion_board/status", Bool, self.status_callback
        )
        rospy.loginfo(
            "DVL killswitch handler initialized, waiting for propulsion board status transitions."
        )

    def status_callback(self, msg):
        current_status = msg.data

        if self.previous_status and not current_status:
            rospy.logwarn("killswitch. disabling DVL.")
            try:
                req = SetBoolRequest(data=False)
                res = self.dvl_enable_service(req)
                if res.success:
                    rospy.loginfo(f"DVL disabled.")
                else:
                    rospy.logerr(f"Failed to disable DVL: {res.message}")
            except Exception as e:
                rospy.logerr(f"Failed to disable DVL: {e}")

        self.previous_status = current_status


if __name__ == "__main__":
    rospy.init_node("dvl_killswitch_handler")
    try:
        node = DvlKillswitchHandler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

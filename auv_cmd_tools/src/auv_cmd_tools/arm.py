import rospy
from std_srvs.srv import SetBool, SetBoolRequest, SetBoolResponse
from std_msgs.msg import Bool
import argparse


class ArmingClient:
    """
    Auv Arm/Disarm Client
    """

    def __init__(self, srv_topic="/turquoise/set_arming", msg_topic="/turquoise/is_armed", timeout=1.5):
        self.srv_topic = srv_topic
        self.msg_topic = msg_topic
        self.connected = False
        self.timeout = timeout
        self.__is_armed = None
        rospy.Subscriber(self.msg_topic, Bool, self.__armed_cb)

    def __connect_service(self):
        try:
            self.connected = True
            rospy.wait_for_service(
                self.srv_topic, rospy.Duration(self.timeout))
        except rospy.ROSException as e:
            self.connected = False
            rospy.logerr(
                "Unable to contact service: {s}".format(s=self.srv_topic))
        self.service = rospy.ServiceProxy(self.srv_topic, SetBool)

    def __armed_cb(self, msg):
        self.__is_armed = msg.data

    def is_armed(self):
        try:
            self.connected = True
            rospy.wait_for_message(self.msg_topic, Bool, timeout=self.timeout)
        except rospy.ROSException as e:
            self.connected = False
            rospy.logerr(
                "Unable to contact service: {s}".format(s=self.srv_topic))
        return self.__is_armed

    def arm(self):
        return self.set_arm(True)

    def disarm(self):
        return self.set_arm(False)

    def set_arm(self, arm):
        self.__connect_service()
        if not self.connected:
            return False

        if type(arm) is not bool:
            rospy.logerr("Argument arm must be boolean type.")
            return False
        req = SetBoolRequest()
        req.data = arm
        resp = self.service.call(req)
        return resp.success

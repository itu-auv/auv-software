import smach
import rospy
from std_srvs.srv import Trigger, TriggerRequest

class FollowPipeState(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=["succeeded", "preempted", "aborted"])
        
        self.start_service_name = "pipe_follower_enhanced/start"
        self.stop_service_name = "pipe_follower_enhanced/stop"
        
        self.start_proxy = rospy.ServiceProxy(self.start_service_name, Trigger)
        self.stop_proxy = rospy.ServiceProxy(self.stop_service_name, Trigger)

    def execute(self, userdata):
        rospy.loginfo("[FollowPipeState] Executing...")
        
        try:
            rospy.loginfo(f"[FollowPipeState] Waiting for {self.start_service_name}...")
            self.start_proxy.wait_for_service(timeout=5.0)
            res = self.start_proxy(TriggerRequest())
            if not res.success:
                rospy.logerr(f"[FollowPipeState] Failed to start: {res.message}")
                return "aborted"
            rospy.loginfo(f"[FollowPipeState] Started: {res.message}")
        except rospy.ROSException as e:
            rospy.logerr(f"[FollowPipeState] Service not available: {e}")
            return "aborted"
        except rospy.ServiceException as e:
            rospy.logerr(f"[FollowPipeState] Service call failed: {e}")
            return "aborted"

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.preempt_requested():
                rospy.loginfo("[FollowPipeState] Preempted. Stopping...")
                try:
                    self.stop_proxy(TriggerRequest())
                except Exception as e:
                    rospy.logerr(f"[FollowPipeState] Error calling stop service: {e}")
                self.service_preempt()
                return "preempted"
            rate.sleep()
            
        return "succeeded"

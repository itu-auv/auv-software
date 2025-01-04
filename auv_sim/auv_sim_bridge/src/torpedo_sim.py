#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger, TriggerResponse
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
from geometry_msgs.msg import Wrench

def launch_torpedo_callback(req):
    """
    Callback function to handle the torpedo launch service.
    """
    rospy.loginfo("Launching torpedo...")

    # Call Gazebo's apply_body_wrench service
    try:
        rospy.loginfo("Waiting for /gazebo/apply_body_wrench service...")
        rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=5.0)
        rospy.loginfo("Service /gazebo/apply_body_wrench found!")
        
        rospy.loginfo("Creating service proxy...")
        apply_wrench = rospy.ServiceProxy('/gazebo/apply_body_wrench', ApplyBodyWrench)
        rospy.loginfo("Service proxy created successfully!")

        # Create a wrench (force and torque) to apply
        wrench = Wrench()
        wrench.force.x = 5.0  # Adjust force in X direction (m/s^2)
        wrench.force.y = 0.0
        wrench.force.z = 0.0

        # Create the service request
        request = ApplyBodyWrenchRequest()
        request.body_name = "/taluy/base_link/left_torpedo"   # Adjust the link name to match your xacro
        request.reference_frame = "world"
        request.wrench = wrench
        request.duration = rospy.Duration(1.0)  # Apply force for 1 second

        # Call the service
        response = apply_wrench(request)
        if response.success:
            rospy.loginfo("Torpedo launched successfully!")
            return TriggerResponse(success=True, message="Torpedo launched successfully!")
        else:
            rospy.logerr("Failed to launch torpedo.")
            return TriggerResponse(success=False, message="Failed to launch torpedo.")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return TriggerResponse(success=False, message=f"Service call failed: {e}")

def main():
    rospy.init_node('launch_torpedo_service')
    rospy.Service('/taluy/actuators/torpedo_1/launch', Trigger, launch_torpedo_callback)
    rospy.loginfo("Torpedo launch service is ready.")
    rospy.spin()

if __name__ == '__main__':
    main()

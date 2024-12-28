#!/usr/bin/env python3

from actionlib import SimpleActionServer
import rospy
import actionlib
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from auv_msgs.msg import FollowPathAction, FollowPathFeedback, FollowPathResult, FollowPathGoal
from auv_msgs.srv import AlignFrame, AlignFrameResponse
import tf2_ros

from navigation_utils.path_utils import (
    create_path_from_frame, get_current_pose, 
    calculate_carrot_pose, broadcast_carrot_frame
)
from navigation_utils.align_frame_utils import AlignFrameController

class FollowPathActionServer:
    def __init__(self):
        rospy.init_node('follow_path_action_server')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.carrot_distance = rospy.get_param('~carrot_distance', 1.0)
        self.goal_position_threshold = rospy.get_param('~goal_position_threshold', 0.1)
        self.goal_angle_threshold = rospy.get_param('~goal_angle_threshold', 0.1)
        self.source_frame = rospy.get_param('~source_frame', "taluy/base_link")
        self.carrot_frame = rospy.get_param('~carrot_frame', "carrot")
        self.control_rate = rospy.Rate(rospy.get_param('~control_rate', 10))
        self.last_target_frame = None
        self.frame_controller = AlignFrameController(
            max_linear_velocity=rospy.get_param('~max_linear_velocity', 0.8),
            max_angular_velocity=rospy.get_param('~max_angular_velocity', 0.9),
            kp=rospy.get_param('~kp', 0.55),
            angle_kp=rospy.get_param('~angle_kp', 0.2)
        )

        self.path_pub = rospy.Publisher('path', Path, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('taluy/cmd_vel', Twist, queue_size=1) #TODO change to cmd_vel after given a namespace in start.launch
        
        self.killswitch_sub = rospy.Subscriber(
            "/taluy/propulsion_board/status",
            Bool,
            self.killswitch_callback,
        )
        
        rospy.Service(
            "align_frame",
            AlignFrame,
            self.handle_align_frame_request
        )
        rospy.logdebug("AlignFrame Service is ready.")
        
        self.action_client = actionlib.SimpleActionClient(
            "follow_path", FollowPathAction)

        
        self.server = actionlib.SimpleActionServer(
            'follow_path',
            FollowPathAction,
            self.execute,
            auto_start=False
        )
        self.server.start()
        rospy.logdebug("FollowPath Action Server is ready.")
        self.action_client.wait_for_server()
        rospy.logdebug("Internal Action Client connected to the FollowPath Action Server.")
        
    def killswitch_callback(self, msg):
        if not msg.data:
            if self.server.is_active():
                self.action_client.cancel_goal() 
                rospy.logwarn("Control canceled")

    def handle_align_frame_request(self, req):
        response = AlignFrameResponse()
        
        path = None # clear any paths
        goal = FollowPathGoal()
        goal.target_frame = req.target_frame
        
        if req.do_planning:
            try:
                path = create_path_from_frame(
                    self.tf_buffer,
                    self.source_frame,
                    req.target_frame,
                    req.angle_offset,
                    #TODO (somebody) add req.keep_orientation
                )
                if path and path.poses:
                    response.success = True
                    response.message = "Valid path created"
                    goal.path = path
                else:
                    rospy.logwarn("Path creation failed")
                    response.success = False
                    response.message = "Path creation failed. Starting direct alignment"
                
            except Exception as e:
                rospy.logerr(f"Error while creating path: {e}")
        
        else:
            response.success = True
            response.message = "Won't create path. Starting direct alignment"
            
        self.action_client.send_goal(goal)
        rospy.logdebug("Goal sent to the Action Server successfully.")
        return response

    def do_alignment(self, path=None, target_frame=None):
        if not path and not target_frame:
            rospy.logerr("Both path and target_frame are None. Cannot perform alignment.")
            return False
            
        try:
            feedback = FollowPathFeedback()
            
            while not rospy.is_shutdown(): # try to succeed until rospy is shutdown
                
                if self.server.is_preempt_requested():
                    
                    # publish stop command for 2 seconds to ensure we stop. 
                    start_time = rospy.get_time()
                    while rospy.get_time() - start_time < 2.0:
                        stop_cmd = Twist()
                        self.cmd_vel_pub.publish(stop_cmd)
                        self.control_rate.sleep()
                    self.server.set_preempted() # then kill.
                    rospy.logwarn("Alignment preempted (do_alignment)")
                    return False
                
                self.frame_controller.enable_alignment()
                # 1. If path is valid, do path-based alignment
                if path and path.poses:
                    self.path_pub.publish(path)
                    current_pose = get_current_pose(self.tf_buffer, self.source_frame)
                    if current_pose is None: 
                        rospy.logwarn("Failed to get current pose. Retrying...")
                        self.control_rate.sleep()
                        continue
                    
                    carrot_pose = calculate_carrot_pose(path, current_pose, self.carrot_distance)
                    broadcast_carrot_frame(self.tf_broadcaster, self.tf_buffer, self.source_frame, carrot_pose)
                    
                    trans, rot = self.frame_controller.get_transform(
                        self.tf_buffer,
                        self.source_frame,
                        self.carrot_frame, 
                    )
                    if trans is None or rot is None: 
                        rospy.logwarn(f"Failed to get transform between {self.source_frame} and {target_frame}")
                        self.control_rate.sleep() 
                        continue
                    
                    if self.frame_controller.is_aligned(trans, rot, self.goal_position_threshold, self.goal_angle_threshold):
                        rospy.loginfo("Goal reached via path following!")
                        return True
                    
                    cmd_vel = self.frame_controller.compute_cmd_vel(trans, rot)
                    self.cmd_vel_pub.publish(cmd_vel)
                    
                    feedback.carrot_pose = carrot_pose #TODO (somebody) position of carrot as feedback is probably enough.
                    feedback.distance_to_goal = np.sqrt(
                        (path.poses[-1].pose.position.x - current_pose.pose.position.x) ** 2 +
                        (path.poses[-1].pose.position.y - current_pose.pose.position.y) ** 2 +
                        (path.poses[-1].pose.position.z - current_pose.pose.position.z) ** 2
                    )
                    self.server.publish_feedback(feedback)
                        
                # 2. If path is not valid, do direct alignment
                elif path is None: 
                    
                    trans, rot = self.frame_controller.get_transform(
                        self.tf_buffer,
                        self.source_frame,
                        target_frame)

                    if self.frame_controller.is_aligned(trans, rot, self.goal_position_threshold, self.goal_angle_threshold):
                        rospy.loginfo("Goal reached via direct alignment!") 
                        return True
                    
                    cmd_vel = self.frame_controller.compute_cmd_vel(trans, rot)
                    self.cmd_vel_pub.publish(cmd_vel)

                self.control_rate.sleep()

            return False # If exited the loop, success is False
        except Exception as e:
            rospy.logerr(f"Error during alignment: {e}") 
            return False

    def execute(self, goal):
        self.preempt_requested = False  # Reset cancel flag when startin a new goal.
        rospy.loginfo("Received a new FollowPath goal.")
        
        if goal.path and goal.path.poses:
            self.path_pub.publish(goal.path)
            
        success = self.do_alignment(path= goal.path if goal.path and goal.path.poses else None,
                                    target_frame=goal.target_frame)
        
        if success:
            result= FollowPathResult(success = True)
            self.server.set_succeeded(result) 
            
        else: 
            rospy.logwarn("FollowPath action failed or was preempted.") 
            result = FollowPathResult(success = False)
            self.server.set_aborted(result) 

        

if __name__ == "__main__":
    FollowPathActionServer()
    rospy.spin()

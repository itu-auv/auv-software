#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist
from auv_msgs.srv import AlignFrameController, AlignFrameControllerResponse
import auv_common_lib.control.enable_state as enable_state
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import angles
import tf2_ros
from path_planner import PathPlanner
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
import math


class FrameAligner:
    def __init__(self):
        rospy.init_node("frame_aligner_node")
        rospy.loginfo("Frame aligner node started")
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.active = False
        self.source_frame = ""
        self.target_frame = ""
        self.angle_offset = 0.0

        self.max_linear_velocity = 0.8
        self.max_angular_velocity = 0.9

        # Initialize the enable signal handler with a timeout duration
        self.rate = rospy.get_param("~rate", 0.1)

        self.enable_pub = rospy.Publisher(
            "enable",
            Bool,
            queue_size=1,
        )

        self.killswitch_sub = rospy.Subscriber(
            "/taluy/propulsion_board/status",
            Bool,
            self.killswitch_callback,
        )

        # Service for setting frames and starting alignment
        rospy.Service(
            "frame_alignment_controller",
            AlignFrameController,
            self.handle_align_request,
        )

        # Service for canceling control
        rospy.Service("cancel_control", Trigger, self.handle_cancel_request)

        self.path_planner = PathPlanner()
        self.current_path = None
        self.current_path_index = 0
        self.carrot_distance = 1.0  # lookahead distance

        self.marker_pub = rospy.Publisher("carrot_marker", Marker, queue_size=10)
        self.path_pub = rospy.Publisher("planned_path", Path, queue_size=1)
        self.closest_index = 0
    def killswitch_callback(self, msg):
        if not msg.data:
            self.active = False

    def handle_align_request(self, req):
        self.source_frame = req.source_frame
        self.target_frame = req.target_frame
        self.angle_offset = req.angle_offset
        self.keep_orientation = req.keep_orientation
        
        self.active = True
        # create path from source to target frame
        path = self.path_planner.get_straight_path(self.source_frame, self.target_frame, self.angle_offset)
        if path is None:
            rospy.logwarn("Failed to get path from path planner, proceeding with simple alignment")
            self.current_path = None
        else:
            rospy.loginfo("Path obtained from path planner")
            self.current_path = path
        return AlignFrameControllerResponse(success=True, message="Alignment started")
    
    def handle_cancel_request(self, req):
        self.active = False
        self.current_path = None
        rospy.loginfo("Control canceled")
        return TriggerResponse(success=True, message="Control deactivated")
    
    """def handle_set_path_request(self, req):)"""
    


    def find_carrot_position(self):
        if not self.current_path or not self.current_path.poses:
            return None
        if len(self.current_path.poses) == 1:
            return np.array([self.current_path.poses[0].pose.position.x,
                            self.current_path.poses[0].pose.position.y,
                            self.current_path.poses[0].pose.position.z])
        
        try: 
            transform = self.tf_buffer.lookup_transform(
                self.current_path.header.frame_id, # frame of where the path is defined (target_frame)
                self.source_frame,
                rospy.Time(0)
            )
            robot_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF Error in find_carrot_point: {str(e)}")
            return None
        
        # 1. find the closest index
        min_dist = float("inf")
        closest_index = self.current_path_index 
        for i in range(self.current_path_index, len(self.current_path.poses)):
            pose = self.current_path.poses[i]
            
            # convert pose position to numpy array for easier math
            point = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z
            ])
            dist = np.linalg.norm(point - robot_pos) # calculate euclidean distance
            
            if dist < min_dist: # if closer than anything we have seen so far
                min_dist = dist
                closest_index = i

        # 2. compute carrot position from closest index
        carrot_distance = self.carrot_distance
        distance_accum = 0.0
        
        for j in range(closest_index, len(self.current_path.poses) -1):
            p_j = self.current_path.poses[j].pose.position
            p_j1 = self.current_path.poses[j+1].pose.position
            segment_vector = np.array([p_j1.x - p_j.x,
                                        p_j1.y - p_j.y,
                                        p_j1.z - p_j.z])
            segment_dist = np.linalg.norm(segment_vector)
            
            if segment_dist < 1e-5: # very small segment? skip.
                continue
            
            if distance_accum + segment_dist >= carrot_distance:
                # Carrot lies on this segment
                remain = carrot_distance - distance_accum
                alpha = remain / segment_dist
                
                carrot_pos = np.array([
                    p_j.x + alpha * (p_j1.x - p_j.x),
                    p_j.y + alpha * (p_j1.y - p_j.y),
                    p_j.z + alpha * (p_j1.z - p_j.z)])
                
                self.current_path_index = j
                
                return carrot_pos
            
            distance_accum += segment_dist
            
        # If we reach here, the carrot is outside all segments.
        return np.array([
            self.current_path.poses[-1].pose.position.x,
            self.current_path.poses[-1].pose.position.y,
            self.current_path.poses[-1].pose.position.z])

    def constrain(self, value, max_value):
        if value > max_value:
            return max_value
        if value < -max_value:
            return -max_value
        return value

    def compute_cmd_vel(self, trans, rot):
        kp = 0.55
        angle_kp = 0.45

        twist = Twist()
        # Set linear velocities based on translation differences
        twist.linear.x = self.constrain(trans[0] * kp, self.max_linear_velocity)
        twist.linear.y = self.constrain(trans[1] * kp, self.max_linear_velocity)

        # Convert quaternion to Euler angles and set angular velocity
        _, _, yaw = euler_from_quaternion(rot)
        twist.angular.z = self.constrain(yaw * angle_kp, self.max_angular_velocity)

        return twist

    def get_transform(self, carrot_pos):
        rate = rospy.Rate(self.rate)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.current_path.header.frame_id,
                self.source_frame,
                rospy.Time(0)
            )
            robot_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            robot_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            _, _, yaw = euler_from_quaternion(robot_quat)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF Error in run loop: {str(e)}")
            rate.sleep()
            
        # Compute translation error in XY plane
        dx = carrot_pos[0] - robot_pos[0]
        dy = carrot_pos[1] - robot_pos[1]
        # dz = carrot_pos[2] - robot_pos[2] # If you need vertical control
        
        # Compute desired heading: angle to carrot
        desired_yaw = math.atan2(dy, dx)
        yaw_error = angles.shortest_angular_distance(yaw, desired_yaw)
        
        # Create a rotation error quaternion from yaw_error
        rot_error_quat = quaternion_from_euler(0, 0, yaw_error)

        trans_error = (dx, dy, 0.0)
        
        return trans_error, rot_error_quat
    
    
    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():

            if not self.active:
                self.enable_pub.publish(Bool(data=False))
                rate.sleep()
                continue

            self.enable_pub.publish(Bool(data=True))

            if self.current_path is None:
                # default to simple alignment if no path is available
                continue
            carrot_pos = self.find_carrot_position()
            if carrot_pos is None:
                # default to simple alignment if no path is available
                continue
            
            trans_error, rot_error_quat = self.get_transform(carrot_pos)
            twist = self.compute_cmd_vel(trans_error, rot_error_quat)
            self.cmd_vel_pub.publish(twist)
            self.path_pub.publish(self.current_path)  # Publish the path for debugging
            rate.sleep()

if __name__ == "__main__":
    try:
        FrameAligner().run()
    except rospy.ROSInterruptException:
        pass
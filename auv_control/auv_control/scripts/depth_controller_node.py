#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseStamped
from auv_msgs.srv import SetDepth, SetDepthRequest, SetDepthResponse
import tf2_geometry_msgs
import tf2_ros


class DepthControllerNode:
    def __init__(self):
        rospy.init_node("depth_controller_node")
        rospy.loginfo("DepthControllerNode has started.")
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Initialize subscribers
        self.odometry_sub = rospy.Subscriber(
            "/taluy/odometry", Odometry, self.odometry_callback
        )
        self.set_depth_service = rospy.Service(
            "/taluy/set_depth", SetDepth, self.target_depth_handler
        )

        # Initialize publisher
        self.cmd_pose_pub = rospy.Publisher("/taluy/cmd_pose", Pose, queue_size=10)

        # Initialize internal state
        self.target_depth = 0.0
        self.current_depth = 0.0

        # Parameters
        self.base_frame = rospy.get_param("~base_link", "taluy/base_link")  # Default to "taluy/base_link" if not set
        self.update_rate = rospy.get_param("~update_rate", 10)
        self.tau = rospy.get_param("~tau", 2.0)
        self.dt = 1.0 / self.update_rate
        self.alpha = self.dt / (self.tau + self.dt)
        # Set up a timer to call the control_loop method at a rate defined by update_rate
        rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.control_loop)

    def target_depth_handler(self, req: SetDepthRequest) -> SetDepthResponse:
        rospy.loginfo(f"Received SetDepth request with target depth: {req.target_depth} and frame_id: {req.frame_id}")
    
    # Check if frame_id is provided
        if not req.frame_id:
            rospy.logwarn("No frame_id provided in SetDepth request. Using target depth directly.")
            self.target_depth = req.target_depth
            return SetDepthResponse(success=True, message="Depth set successfully without transformation")

        try:
            # Look up the transform from the supplied frame to the base frame
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,  # Target frame (controller's base frame)
                req.frame_id,     # Source frame (supplied frame_id)
                rospy.Time(0),    # current time
                rospy.Duration(1.0)  # Timeout after 1 second
            )
            rospy.loginfo(f"Transform from {req.frame_id} to {self.base_frame} obtained successfully.")

            # Log the translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            rospy.loginfo(f"Translation: x={translation.x}, y={translation.y}, z={translation.z}")
            rospy.loginfo(f"Rotation: x={rotation.x}, y={rotation.y}, z={rotation.z}, w={rotation.w}")
            
            
            # Create a PoseStamped message in the source frame
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = req.frame_id
            pose_stamped.pose.position.z = req.target_depth
            pose_stamped.pose.orientation.w = 1.0

            # Transform the pose to the base frame
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)

            # Update the target depth with the transformed z position
            self.target_depth = transformed_pose.pose.position.z
            rospy.loginfo(f"Target depth transformed to base frame: {self.target_depth}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Transform error: %s", e)
            return SetDepthResponse(success=False, message=str(e))

        return SetDepthResponse(success=True, message="Depth set successfully")


    def odometry_callback(self, msg):
        self.current_depth = msg.pose.pose.position.z
        rospy.loginfo(f"Current depth updated: {self.current_depth}")
        
    def control_loop(self, event):
        # Create and publish the cmd_pose message        
        cmd_pose = Pose()
        cmd_pose.position.z = self.target_depth
        cmd_pose.orientation.w = 1.0
        self.cmd_pose_pub.publish(cmd_pose)
        rospy.loginfo(f"Published cmd_pose with target depth: {self.target_depth}")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        depth_controller_node = DepthControllerNode()
        depth_controller_node.run()
    except rospy.ROSInterruptException:
        pass
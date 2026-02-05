#!/usr/bin/env python3
"""
ROS node for MegaPose 6D pose estimation.
Subscribes to camera topics, sends to MegaPose ZMQ server, publishes poses.
"""

import rospy
import zmq
import json
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, TransformStamped
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, TransformStamped
from std_msgs.msg import Header
from ultralytics_ros.msg import YoloResult
import cv2
import tf2_ros


class MegaPoseNode:
    def __init__(self):
        rospy.init_node("megapose_node")
        
        # TF Broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Parameters
        self.zmq_host = rospy.get_param("~zmq_host", "localhost")
        self.zmq_port = rospy.get_param("~zmq_port", 5556)
        self.object_label = rospy.get_param("~object_label", "")
        self.rate_hz = rospy.get_param("~rate", 10.0)
        self.force_frame_id = rospy.get_param("~force_frame_id", "")
        self.camera_name = rospy.get_param("~camera_name", "") # Used for TF suffix
        
        if not self.object_label:
            rospy.logerr("object_label parameter is required!")
            return
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be updated from CameraInfo)
        self.camera_K = None
        self.camera_frame_id = None
        
        # Latest image
        self.latest_image = None
        self.latest_bbox = None  # [x1, y1, x2, y2]
        
        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        server_address = f"tcp://{self.zmq_host}:{self.zmq_port}"
        self.socket.connect(server_address)
        rospy.loginfo(f"Connected to MegaPose server at {server_address}")
        
        # Publishers
        # self.pose_pub = rospy.Publisher("~poses", PoseArray, queue_size=10)
        # self.pose_stamped_pub = rospy.Publisher("~pose", PoseStamped, queue_size=10)
        
        # Subscribers
        rospy.Subscriber("~image", Image, self.image_callback)
        rospy.Subscriber("~camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("~bbox", YoloResult, self.bbox_callback)
        
        rospy.loginfo(f"MegaPose ROS node initialized for label: {self.object_label}")
    
    def bbox_callback(self, msg):
        """Update bbox from external source (ultralytics_ros/YoloResult)."""

        detections_list = msg.detections.detections
        
        # We take the detection with the highest confidence if multiple are present
        if detections_list:
            best_detection = detections_list[0]
            if len(detections_list) > 1:
                try:
                    best_detection = max(detections_list, 
                                       key=lambda d: d.results[0].score if d.results else -1.0)
                except Exception:
                    pass

            # Convert BoundingBox2D (center_x, center_y, size_x, size_y) to [x1, y1, x2, y2]
            cx = best_detection.bbox.center.x
            cy = best_detection.bbox.center.y
            w = best_detection.bbox.size_x
            h = best_detection.bbox.size_y
            
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            
            self.latest_bbox = [x1, y1, x2, y2]
        else:
            self.latest_bbox = None
    
    def camera_info_callback(self, msg):
        """Update camera intrinsics from CameraInfo."""
        self.camera_K = np.array(msg.K).reshape(3, 3).tolist()
        # Use forced frame id if set, otherwise use msg header
        if self.force_frame_id:
            self.camera_frame_id = self.force_frame_id
        else:
            self.camera_frame_id = msg.header.frame_id
    
    def image_callback(self, msg):
        """Store latest image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
    
    def run_inference(self):
        """Send image to MegaPose server and get poses."""
        if self.latest_image is None:
            return None
        
        if self.camera_K is None:
            rospy.logwarn_throttle(5, "Waiting for camera_info...")
            return None
        
        # Encode image
        _, image_bytes = cv2.imencode(".jpg", cv2.cvtColor(self.latest_image, cv2.COLOR_RGB2BGR))
        image_bytes = image_bytes.tobytes()
        
        # Prepare metadata (only label + camera, mesh is loaded on server)
        metadata = {
            "label": self.object_label,
            "camera_K": self.camera_K,
        }
        
        # Add bbox if available
        if self.latest_bbox is not None:
            metadata["bbox"] = self.latest_bbox
        
        # Prepare frames: [metadata, image]
        frames = [json.dumps(metadata).encode("utf-8"), image_bytes]
        
        try:
            # Send request
            self.socket.send_multipart(frames)
            
            # Receive response
            response_bytes = self.socket.recv()
            response = json.loads(response_bytes.decode("utf-8"))
            
            if "error" in response:
                rospy.logerr(f"MegaPose error: {response['error']}")
                return None
            
            return response.get("poses", [])
        
        except zmq.Again:
            rospy.logwarn("MegaPose server timeout")
            # Reconnect socket
            self.socket.close()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, 5000)
            self.socket.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
            return None
        except Exception as e:
            rospy.logerr(f"ZMQ error: {e}")
            return None

    def publish_poses(self, poses):
        """Publish poses (TF only)."""
        if not poses:
            return
        
        header = Header()
        header.stamp = rospy.Time.now()
        
        if self.force_frame_id:
            header.frame_id = self.force_frame_id
        elif self.camera_frame_id:
            header.frame_id = self.camera_frame_id
        else:
            header.frame_id = "camera_color_optical_frame"
        
        pose_array = PoseArray()
        pose_array.header = header

        for pose_data in poses:
            pose = Pose()
            
            # Quaternion (wxyz -> xyzw for ROS)
            q = pose_data["quaternion"]
            pose.orientation.w = q[0]
            pose.orientation.x = q[1]
            pose.orientation.y = q[2]
            pose.orientation.z = q[3]
            
            # Translation
            t = pose_data["translation"]
            pose.position.x = t[0]
            pose.position.y = t[1]
            pose.position.z = t[2]
            
            pose_array.poses.append(pose)
        
        # Broadcast TF for the first pose
        if pose_array.poses:
             self.broadcast_tf(pose_array.poses[0], header)
            
    def broadcast_tf(self, pose, header):
        """Broadcast transform for the object."""
        t = TransformStamped()
        
        t.header.stamp = header.stamp
        t.header.frame_id = header.frame_id
        
        # Unique TF frame for each camera's detection to avoid fighting
        suffix = f"_{self.camera_name}" if self.camera_name else ""
        t.child_frame_id = f"{self.object_label}{suffix}_link"
        
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        
        t.transform.rotation.x = pose.orientation.x
        t.transform.rotation.y = pose.orientation.y
        t.transform.rotation.z = pose.orientation.z
        t.transform.rotation.w = pose.orientation.w
        
        self.tf_broadcaster.sendTransform(t)
    
    def spin(self):
        """Main loop."""
        rate = rospy.Rate(self.rate_hz)
        
        while not rospy.is_shutdown():
            poses = self.run_inference()
            if poses:
                self.publish_poses(poses)
                rospy.loginfo_throttle(1, f"Published {len(poses)} pose(s)")
            
            rate.sleep()
        
        # Cleanup
        self.socket.close()
        self.context.term()


def main():
    try:
        node = MegaPoseNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import rospy
import cv2 as cv
from cv2 import aruco
import numpy as np
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import tf2_ros
import tf
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf2_geometry_msgs
import auv_common_lib.vision.camera_calibrations as camera_calibrations


class ArucoDepthEstimator:
    def __init__(self):
        rospy.init_node('aruco_depth_estimator', anonymous=True)

        # Get parameters
        self.camera_namespace = 'cameras/cam_front'
        self.camera_frame = 'front_camera_optical_link'
        self.odom_frame = 'odom'
        self.image_topic = 'camera/image_raw'
        self.marker_size = rospy.get_param('~marker_size', 0.05)  # meters

        # Get Camera calibration
        try:
            self.cam_calib = camera_calibrations.CameraCalibrationFetcher(
                self.camera_namespace, True
            ).get_camera_info()
            self.cam_mat = np.array(self.cam_calib.K).reshape(3, 3)
            self.dist_coef = np.array(self.cam_calib.D)
            rospy.loginfo(f"  cx={self.cam_mat[0,2]:.2f}, cy={self.cam_mat[1,2]:.2f}")
        except Exception as e:
            rospy.logerr(f"Failed to load camera calibration: {e}")
            raise

        # ArUco setup
        self.marker_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.param_markers = aruco.DetectorParameters_create()
        
        # Optimize detector parameters for better detection (tunable via ROS params)
        self.param_markers.adaptiveThreshConstant = rospy.get_param('~adaptiveThreshConstant', 7)
        self.param_markers.minMarkerPerimeterRate = rospy.get_param('~minMarkerPerimeterRate', 0.03)
        self.param_markers.maxMarkerPerimeterRate = rospy.get_param('~maxMarkerPerimeterRate', 4.0)

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher(
            '~image_annotated/compressed', 
            CompressedImage, 
            queue_size=1
        )
        self.image_sub = rospy.Subscriber(
            self.image_topic, 
            Image, 
            self.image_callback,
            queue_size=1,
            buff_size=2**24  # 16MB buffer for large images
        )
        
        # TF2 setup
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        
        rospy.loginfo("ArUco Depth Estimator initialized successfully")

    def image_callback(self, msg):
        self.frame_count += 1
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
            return

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        marker_corners, marker_IDs, rejected = aruco.detectMarkers(
            gray_frame, self.marker_dict, parameters=self.param_markers
        )

        if marker_IDs is not None and len(marker_IDs) > 0:
            self.detection_count += len(marker_IDs)
            
            # Estimate pose of each marker
            rVecs, tVecs, _ = aruco.estimatePoseSingleMarkers(
                marker_corners, self.marker_size, self.cam_mat, self.dist_coef
            )

            # Draw detected markers on the frame
            aruco.drawDetectedMarkers(frame, marker_corners, marker_IDs)
            
            # Process each detected marker
            for i in range(len(marker_IDs)):
                marker_id = marker_IDs[i][0]
                rVec = rVecs[i][0]
                tVec = tVecs[i][0]
                
                # Draw axis for visualization
                aruco.drawAxis(
                    frame, self.cam_mat, self.dist_coef, 
                    rVec, tVec, self.marker_size * 0.5
                )
                
                # Calculate distance
                distance = np.linalg.norm(tVec)
                
                # Draw distance text on frame
                corner = marker_corners[i][0][0]  # Top-left corner
                text = f"ID:{marker_id} D:{distance:.2f}m"
                cv.putText(
                    frame, text, 
                    (int(corner[0]), int(corner[1]) - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                
                # Publish TF transform
                self.publish_marker_transform(marker_id, rVec, tVec, msg.header)
                
                # Log detection
                rospy.logdebug(
                    f"Marker {marker_id}: distance={distance:.3f}m, "
                    f"position=({tVec[0]:.3f}, {tVec[1]:.3f}, {tVec[2]:.3f})"
                )
        else:
            # Draw rejected markers for debugging
            if rejected is not None and len(rejected) > 0:
                aruco.drawDetectedMarkers(frame, rejected, borderColor=(100, 0, 255))
        
        # Add frame info
        info_text = f"Frame: {self.frame_count} | Markers: {len(marker_IDs) if marker_IDs is not None else 0}"
        cv.putText(
            frame, info_text, (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Publish annotated image
        self.publish_annotated_image(frame, msg.header.stamp)

    def publish_marker_transform(self, marker_id, rVec, tVec, header):
        """Publish TF transform for detected marker in odom frame"""
        try:
            # Create pose in camera frame
            marker_pose = PoseStamped()
            marker_pose.header.frame_id = self.camera_frame
            marker_pose.header.stamp = header.stamp
            
            # Set position
            marker_pose.pose.position.x = tVec[0]
            marker_pose.pose.position.y = tVec[1]
            marker_pose.pose.position.z = tVec[2]
            
            # Convert rotation vector to quaternion
            rotation_matrix = np.eye(4)
            rotation_matrix[0:3, 0:3], _ = cv.Rodrigues(rVec)
            quaternion = tf.transformations.quaternion_from_matrix(rotation_matrix)
            
            marker_pose.pose.orientation.x = quaternion[0]
            marker_pose.pose.orientation.y = quaternion[1]
            marker_pose.pose.orientation.z = quaternion[2]
            marker_pose.pose.orientation.w = quaternion[3]

            # Transform to odom frame
            try:
                pose_odom = self.tf_buffer.transform(
                    marker_pose, 
                    self.odom_frame, 
                    marker_pose.header.stamp,
                    rospy.Duration(1.0)
                )
                
                # Create and broadcast transform
                t = TransformStamped()
                t.header.stamp = header.stamp
                t.header.frame_id = self.odom_frame
                t.child_frame_id = f"aruco_{marker_id}"
                t.transform.translation.x = pose_odom.pose.position.x
                t.transform.translation.y = pose_odom.pose.position.y
                t.transform.translation.z = pose_odom.pose.position.z
                t.transform.rotation = pose_odom.pose.orientation
                
                self.tf_broadcaster.sendTransform(t)
                
            except (tf2_ros.LookupException, 
                    tf2_ros.ConnectivityException, 
                    tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(
                    5.0, 
                    f"Could not transform marker {marker_id} to odom: {e}"
                )
        except Exception as e:
            rospy.logerr(f"Error publishing transform for marker {marker_id}: {e}")

    def publish_annotated_image(self, frame, stamp):
        """Publish annotated image as CompressedImage"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(frame, dst_format='jpeg')
            compressed_msg.header.stamp = stamp
            self.image_pub.publish(compressed_msg)
            
        except Exception as e:
            rospy.logerr(f"Failed to publish annotated image: {e}")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ArucoDepthEstimator()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        raise
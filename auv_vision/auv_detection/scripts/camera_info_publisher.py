#!/usr/bin/env python3
import os
import rospy
import yaml
from sensor_msgs.msg import CameraInfo

def load_camera_info(yaml_file, camera_name):
    try:
        rospy.loginfo(f"Trying to load calibration for {camera_name} from {yaml_file}")
        with open(yaml_file, "r") as file:
            calib_data = yaml.safe_load(file)
        
        if camera_name not in calib_data:
            rospy.logerr(f"Camera {camera_name} not found in calibration file")
            return None
            
        rospy.loginfo(f"Successfully loaded calibration data for {camera_name}")
        rospy.logdebug(f"Calibration data: {calib_data[camera_name]}")
            
        cam_info_msg = CameraInfo()
        cam_info_msg.width = calib_data[camera_name]["image_width"]
        cam_info_msg.height = calib_data[camera_name]["image_height"]
        cam_info_msg.K = calib_data[camera_name]["camera_matrix"]["data"]
        cam_info_msg.D = calib_data[camera_name]["distortion_coefficients"]["data"]
        cam_info_msg.R = calib_data[camera_name]["rectification_matrix"]["data"]
        cam_info_msg.P = calib_data[camera_name]["projection_matrix"]["data"]
        cam_info_msg.distortion_model = calib_data[camera_name]["distortion_model"]
        return cam_info_msg
    except Exception as e:
        rospy.logerr(f"Error loading camera info: {str(e)}")
        return None

def publish_camera_info():
    rospy.init_node("custom_camera_info_publisher", log_level=rospy.DEBUG)
    rospy.loginfo("Starting camera_info_publisher node")
    
    left_pub = rospy.Publisher("/camera/infra1/camera_info_custom", CameraInfo, queue_size=10)
    right_pub = rospy.Publisher("/camera/infra2/camera_info_custom", CameraInfo, queue_size=10)
    
    rospy.loginfo(f"Created publishers: \n- {left_pub.name}\n- {right_pub.name}")
    
    # Get the package path
    pkg_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(pkg_path, 'config/custom_calibration.yaml')
    
    rospy.loginfo(f"Looking for calibration file at: {yaml_path}")
    
    if not os.path.exists(yaml_path):
        rospy.logerr(f"Calibration file not found at {yaml_path}")
        return
    
    rospy.loginfo("Found calibration file")
        
    left_info = load_camera_info(yaml_path, "camera_infra1")
    right_info = load_camera_info(yaml_path, "camera_infra2")
    
    if left_info is None or right_info is None:
        rospy.logerr("Failed to load camera calibration data")
        return

    rospy.loginfo("Successfully loaded both camera calibrations")
    publish_count = 0
    rate = rospy.Rate(10)  # 10 Hz
    
    while not rospy.is_shutdown():
        # Add current timestamp
        now = rospy.Time.now()
        left_info.header.stamp = now
        right_info.header.stamp = now
        
        left_pub.publish(left_info)
        right_pub.publish(right_info)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_camera_info()
    except rospy.ROSInterruptException:
        pass

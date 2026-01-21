#!/usr/bin/env python3
"""
ROS service node for managing object detection from web GUI
Provides services to start/stop detection with CUDA options
"""

import rospy
import subprocess
import signal
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse


class DetectionService:
    def __init__(self):
        rospy.init_node('web_gui_detection_service', anonymous=False)
        
        self.detect_process = None
        
        # Services for detection control
        self.start_service = rospy.Service(
            '/taluy/web_gui/start_detection',
            SetBool,  # Use SetBool to pass CUDA flag
            self.start_detection
        )
        
        self.stop_service = rospy.Service(
            '/taluy/web_gui/stop_detection',
            Trigger,
            self.stop_detection
        )
        
        rospy.loginfo("Web GUI Detection Service started")
        rospy.loginfo("Available services:")
        rospy.loginfo("  - /taluy/web_gui/start_detection (SetBool - data=true for CUDA)")
        rospy.loginfo("  - /taluy/web_gui/stop_detection")
    
    def start_detection(self, req):
        """Start object detection with optional CUDA"""
        try:
            # Stop any existing detection first
            if self.detect_process is not None:
                rospy.loginfo("Stopping existing detection process first...")
                self.stop_detection(None)
                rospy.sleep(1.0)
            
            # Build command
            cmd = ["roslaunch", "auv_detection", "tracker.launch"]
            
            # Add device argument based on CUDA flag
            if req.data:  # CUDA enabled
                cmd.append("device:=cuda:0")
                device_name = "CUDA (GPU)"
            else:  # CPU only
                cmd.append("device:=cpu")
                device_name = "CPU"
            
            rospy.loginfo("=" * 60)
            rospy.loginfo(f"STARTING OBJECT DETECTION")
            rospy.loginfo(f"Requested Device: {device_name}")
            rospy.loginfo(f"Command: {' '.join(cmd)}")
            rospy.loginfo("=" * 60)
            
            # Launch detection with output visible
            self.detect_process = subprocess.Popen(cmd)
            
            rospy.loginfo(f"Detection process started with PID: {self.detect_process.pid}")
            
            # Wait a moment for nodes to start and then verify the device parameter
            rospy.loginfo("Waiting for tracker node to initialize...")
            rospy.sleep(3.0)
            
            # Try to get the actual device parameter from the tracker node
            try:
                # The parameter is set under the tracker_node namespace
                device_param = rospy.get_param('/tracker_node/device', None)
                if device_param:
                    rospy.loginfo("=" * 60)
                    rospy.loginfo("✓ DEVICE VERIFICATION:")
                    rospy.loginfo(f"  Actual device parameter: {device_param}")
                    if 'cuda' in str(device_param).lower():
                        rospy.loginfo("  ✓✓✓ GPU/CUDA ACCELERATION ACTIVE ✓✓✓")
                    else:
                        rospy.loginfo("  ℹ CPU MODE ACTIVE (No GPU)")
                    rospy.loginfo("=" * 60)
                else:
                    rospy.logwarn("Device parameter not found yet, node may still be starting...")
            except Exception as e:
                rospy.logwarn(f"Could not verify device parameter: {e}")
                rospy.logwarn("Node may still be initializing, check 'rosparam get /tracker_node/device' manually")
            
            return SetBoolResponse(
                success=True,
                message=f"Detection started with {device_name}"
            )
        except Exception as e:
            rospy.logerr(f"Failed to start detection: {str(e)}")
            return SetBoolResponse(
                success=False,
                message=f"Failed to start detection: {str(e)}"
            )
    
    def stop_detection(self, req):
        """Stop object detection"""
        try:
            if self.detect_process is None:
                rospy.logwarn("No detection process to stop")
                return TriggerResponse(
                    success=True,
                    message="No detection process running"
                )
            
            rospy.loginfo("=" * 60)
            rospy.loginfo(f"STOPPING OBJECT DETECTION (PID: {self.detect_process.pid})")
            rospy.loginfo("=" * 60)
            
            # Send SIGINT to roslaunch to properly shutdown all nodes
            self.detect_process.send_signal(signal.SIGINT)
            
            try:
                rospy.loginfo("Waiting for graceful shutdown...")
                self.detect_process.wait(timeout=5)
                rospy.loginfo("Detection stopped gracefully")
            except subprocess.TimeoutExpired:
                rospy.logwarn("Detection did not stop gracefully, forcing termination...")
                self.detect_process.terminate()
                try:
                    self.detect_process.wait(timeout=2)
                    rospy.loginfo("Detection terminated")
                except subprocess.TimeoutExpired:
                    rospy.logerr("Detection did not terminate, killing...")
                    self.detect_process.kill()
                    self.detect_process.wait()
                    rospy.loginfo("Detection killed")
            
            self.detect_process = None
            rospy.loginfo("Detection stopped successfully")
            
            return TriggerResponse(
                success=True,
                message="Detection stopped successfully"
            )
        except Exception as e:
            rospy.logerr(f"Failed to stop detection: {str(e)}")
            self.detect_process = None
            return TriggerResponse(
                success=False,
                message=f"Failed to stop detection: {str(e)}"
            )
    
    def shutdown(self):
        """Clean shutdown"""
        if self.detect_process is not None:
            rospy.loginfo("Cleaning up detection process...")
            self.stop_detection(None)
    
    def run(self):
        """Keep the node running"""
        rospy.on_shutdown(self.shutdown)
        rospy.spin()


if __name__ == '__main__':
    try:
        service = DetectionService()
        service.run()
    except rospy.ROSInterruptException:
        pass

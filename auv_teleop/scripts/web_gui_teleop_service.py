#!/usr/bin/env python3

import rospy
import subprocess
import signal
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

class WebGUITeleopService:
    def __init__(self):
        rospy.init_node('web_gui_teleop_service', anonymous=True)
        
        # Get namespace from parameter server, default to 'taluy'
        self.namespace = rospy.get_param('~namespace', 'taluy')
        
        # Store the teleop process
        self.teleop_process = None
        
        # Create services
        self.start_service = rospy.Service(
            f'/{self.namespace}/web_gui/start_teleop',
            SetBool,
            self.handle_start_teleop
        )
        
        self.stop_service = rospy.Service(
            f'/{self.namespace}/web_gui/stop_teleop',
            Trigger,
            self.handle_stop_teleop
        )
        
        rospy.loginfo(f"Web GUI Teleop Service initialized with namespace: {self.namespace}")
        rospy.loginfo(f"Services available:")
        rospy.loginfo(f"  - /{self.namespace}/web_gui/start_teleop (SetBool - data=true for Xbox)")
        rospy.loginfo(f"  - /{self.namespace}/web_gui/stop_teleop (Trigger)")

    def handle_start_teleop(self, req):
        """Start teleop service. req.data = True for Xbox controller, False for default (joy)"""
        try:
            # Stop existing teleop if running
            if self.teleop_process is not None:
                rospy.logwarn("Teleop already running, stopping it first...")
                self.stop_teleop_process()
            
            # Get device ID from parameter (can be set by web GUI before calling service)
            device_id = rospy.get_param('~device_id', 1)  # Default to js1
            
            # Build the roslaunch command
            controller = "xbox" if req.data else "joy"
            cmd = [
                "roslaunch",
                "auv_teleop",
                "start_teleop.launch",
                f"namespace:={self.namespace}",
                f"controller:={controller}",
                f"id:={device_id}"
            ]
            
            rospy.loginfo(f"Starting teleop with controller: {controller}")
            rospy.loginfo(f"Command: {' '.join(cmd)}")
            rospy.loginfo(f"Using joystick device: /dev/input/js{device_id}")
            
            # Start the process
            self.teleop_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout for easier debugging
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
            )
            
            # Give it a moment to start and check if it's still running
            rospy.sleep(0.5)
            if self.teleop_process.poll() is not None:
                # Process already exited
                stdout, _ = self.teleop_process.communicate()
                error_msg = f"Teleop process failed to start. Output: {stdout.decode()}"
                rospy.logerr(error_msg)
                self.teleop_process = None
                return SetBoolResponse(success=False, message=error_msg)
            
            rospy.loginfo(f"Teleop process started successfully (PID: {self.teleop_process.pid})")
            
            return SetBoolResponse(
                success=True,
                message=f"Teleop started with {controller} controller"
            )
            
        except Exception as e:
            error_msg = f"Failed to start teleop: {str(e)}"
            rospy.logerr(error_msg)
            return SetBoolResponse(success=False, message=error_msg)

    def handle_stop_teleop(self, req):
        """Stop teleop service"""
        try:
            if self.teleop_process is None:
                return TriggerResponse(
                    success=False,
                    message="No teleop process is running"
                )
            
            rospy.loginfo("Stopping teleop...")
            self.stop_teleop_process()
            
            return TriggerResponse(
                success=True,
                message="Teleop stopped successfully"
            )
            
        except Exception as e:
            error_msg = f"Failed to stop teleop: {str(e)}"
            rospy.logerr(error_msg)
            return TriggerResponse(success=False, message=error_msg)

    def stop_teleop_process(self):
        """Stop the teleop process gracefully"""
        if self.teleop_process is None:
            return
        
        try:
            # Try graceful termination first (SIGINT)
            rospy.loginfo("Sending SIGINT to teleop process...")
            self.teleop_process.send_signal(signal.SIGINT)
            
            # Wait up to 2 seconds
            try:
                self.teleop_process.wait(timeout=2)
                rospy.loginfo("Teleop process terminated gracefully")
            except subprocess.TimeoutExpired:
                # If still running, use SIGTERM
                rospy.logwarn("Process did not terminate, sending SIGTERM...")
                self.teleop_process.terminate()
                try:
                    self.teleop_process.wait(timeout=2)
                    rospy.loginfo("Teleop process terminated with SIGTERM")
                except subprocess.TimeoutExpired:
                    # Last resort: SIGKILL
                    rospy.logwarn("Process still running, sending SIGKILL...")
                    self.teleop_process.kill()
                    self.teleop_process.wait()
                    rospy.loginfo("Teleop process killed")
            
            self.teleop_process = None
            
        except Exception as e:
            rospy.logerr(f"Error stopping teleop process: {str(e)}")
            self.teleop_process = None

    def shutdown_hook(self):
        """Clean up on node shutdown"""
        rospy.loginfo("Shutting down Web GUI Teleop Service...")
        if self.teleop_process is not None:
            rospy.loginfo("Stopping teleop process...")
            self.stop_teleop_process()

    def run(self):
        """Main service loop"""
        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo("Web GUI Teleop Service ready")
        rospy.spin()

if __name__ == '__main__':
    try:
        service = WebGUITeleopService()
        service.run()
    except rospy.ROSInterruptException:
        pass

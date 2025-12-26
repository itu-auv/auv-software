#!/usr/bin/env python3
"""
ROS service node for executing system commands from web GUI
Provides a secure interface for web GUI to launch applications like rqt_image_view
"""

import rospy
import subprocess
from std_srvs.srv import Trigger, TriggerResponse


class WebGuiCommandService:
    def __init__(self):
        rospy.init_node('web_gui_command_service', anonymous=False)
        
        # Service for launching rqt_image_view
        self.rqt_service = rospy.Service(
            '/taluy/web_gui/launch_rqt_image_view',
            Trigger,
            self.launch_rqt_image_view
        )
        
        rospy.loginfo("Web GUI Command Service started")
        rospy.loginfo("Available services:")
        rospy.loginfo("  - /taluy/web_gui/launch_rqt_image_view")
    
    def launch_rqt_image_view(self, req):
        """Launch rqt_image_view application"""
        try:
            rospy.loginfo("Launching rqt_image_view...")
            
            # Launch rqt with image view plugin
            # Use shell=True and source ROS environment to ensure proper environment
            import os
            env = os.environ.copy()
            
            # Ensure ROS environment is set
            if 'ROS_MASTER_URI' not in env:
                rospy.logwarn("ROS_MASTER_URI not set in environment")
            
            # Launch in background, detached from this process
            subprocess.Popen(
                ["rqt", "-s", "rqt_image_view"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            rospy.loginfo("rqt_image_view process started")
            
            return TriggerResponse(
                success=True,
                message="rqt_image_view launched successfully"
            )
        except Exception as e:
            rospy.logerr(f"Failed to launch rqt_image_view: {str(e)}")
            return TriggerResponse(
                success=False,
                message=f"Failed to launch rqt_image_view: {str(e)}"
            )
    
    def run(self):
        """Keep the node running"""
        rospy.spin()


if __name__ == '__main__':
    try:
        service = WebGuiCommandService()
        service.run()
    except rospy.ROSInterruptException:
        pass

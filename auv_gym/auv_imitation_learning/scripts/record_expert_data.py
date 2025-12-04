#!/usr/bin/env python3
#TODO this is [WIP] and not ready for use
import rospy
import argparse
import subprocess
import signal
import sys
import os
import threading
import time
from std_srvs.srv import Empty
from auv_smach.gate import NavigateThroughGateState
import smach
import smach_ros

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\nShutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class ExpertRecorder:
    def __init__(self, mode, env, num_episodes, output_dir, bag_prefix="expert_data"):
        self.mode = mode
        self.env = env
        self.num_episodes = num_episodes
        self.output_dir = output_dir
        self.bag_prefix = bag_prefix
        self.bag_process = None
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ROS Init
        rospy.init_node('expert_data_recorder', anonymous=True)
        
        # Services
        if self.env == 'sim':
            rospy.loginfo("Waiting for /gazebo/reset_world service...")
            try:
                rospy.wait_for_service('/gazebo/reset_world', timeout=5.0)
                self.reset_world_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
                rospy.loginfo("Connected to /gazebo/reset_world")
            except rospy.ROSException:
                rospy.logerr("Service /gazebo/reset_world not available! Is Gazebo running?")
                sys.exit(1)

    def start_bag_recording(self, episode_idx=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if episode_idx is not None:
            bag_name = f"{self.bag_prefix}_ep{episode_idx}_{timestamp}.bag"
        else:
            bag_name = f"{self.bag_prefix}_{timestamp}.bag"
            
        bag_path = os.path.join(self.output_dir, bag_name)
        
        # Define topics to record
        # Adjust these topics based on what's needed for BC (e.g., camera, odometry, cmd_vel)
        topics = [
            "/taluy/cmd_vel",
            "/taluy/cameras/cam_front/image_raw/compressed",
            "/taluy/odometry",
            "/tf",
            "/tf_static"
        ]
        
        cmd = ["rosbag", "record", "-O", bag_path] + topics
        
        rospy.loginfo(f"Starting recording: {bag_path}")
        self.bag_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_bag_recording(self):
        if self.bag_process:
            rospy.loginfo("Stopping recording...")
            self.bag_process.send_signal(signal.SIGINT)
            self.bag_process.wait()
            self.bag_process = None

    def reset_sim(self):
        if self.env == 'sim':
            rospy.loginfo("Resetting simulation...")
            try:
                self.reset_world_proxy()
                rospy.sleep(1.0) # Wait for physics to settle
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

    def run_autonomous_episode(self):
        # Create SMACH state machine for the gate task
        sm = smach.StateMachine(outcomes=['succeeded', 'preempted', 'aborted'])
        
        # Get parameters from ROS param server or use defaults
        gate_depth = rospy.get_param("~gate_depth", -1.35)
        gate_search_depth = rospy.get_param("~gate_search_depth", -0.7)
        gate_exit_angle = rospy.get_param("~gate_exit_angle", 0.0)
        
        with sm:
            smach.StateMachine.add(
                'NAVIGATE_THROUGH_GATE',
                NavigateThroughGateState(
                    gate_depth=gate_depth,
                    gate_search_depth=gate_search_depth,
                    gate_exit_angle=gate_exit_angle
                ),
                transitions={
                    'succeeded': 'succeeded',
                    'preempted': 'preempted',
                    'aborted': 'aborted'
                }
            )
            
        outcome = sm.execute()
        return outcome

    def run(self):
        if self.mode == 'autonomous':
            self.run_autonomous_loop()
        elif self.mode == 'human':
            self.run_human_loop()

    def run_autonomous_loop(self):
        for i in range(self.num_episodes):
            if shutdown_requested or rospy.is_shutdown():
                break
                
            rospy.loginfo(f"--- Starting Episode {i+1}/{self.num_episodes} ---")
            
            if self.env == 'sim':
                self.reset_sim()
            
            self.start_bag_recording(episode_idx=i+1)
            
            outcome = self.run_autonomous_episode()
            rospy.loginfo(f"Episode {i+1} finished with outcome: {outcome}")
            
            self.stop_bag_recording()
            
            if outcome == 'preempted':
                break

    def run_human_loop(self):
        rospy.loginfo("--- Human Expert Mode ---")
        if self.env == 'sim':
            rospy.loginfo("Press 'r' + Enter to reset simulation. Press Ctrl+C to exit.")
        else:
            rospy.loginfo("Recording... Press Ctrl+C to stop.")

        self.start_bag_recording()

        # Input listener thread
        def input_listener():
            while not shutdown_requested and not rospy.is_shutdown():
                try:
                    user_input = input()
                    if user_input.strip() == 'r' and self.env == 'sim':
                        # Stop current recording (optional, or just mark it?)
                        # User requested reset
                        # For continuous recording, we might just reset.
                        # But maybe better to split bags?
                        # Let's keep it simple: Continuous recording, just reset sim.
                        # Or: Stop, Reset, Start new bag.
                        # Let's do: Stop, Reset, Start new bag.
                        self.stop_bag_recording()
                        self.reset_sim()
                        self.start_bag_recording()
                except EOFError:
                    break

        if self.env == 'sim':
            t = threading.Thread(target=input_listener)
            t.daemon = True
            t.start()

        # Keep main thread alive
        rate = rospy.Rate(10)
        while not shutdown_requested and not rospy.is_shutdown():
            rate.sleep()
            
        self.stop_bag_recording()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record expert data for AUV")
    parser.add_argument("--mode", default="autonomous", choices=["autonomous", "human"], help="Recording mode")
    parser.add_argument("--env", default="sim", choices=["sim", "real"], help="Environment type")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes (autonomous mode)")
    parser.add_argument("--output_dir", default=os.path.join(os.environ.get("HOME"), "bags/expert_data"), help="Output directory for bags")
    
    args = parser.parse_args(rospy.myargv()[1:])
    
    recorder = ExpertRecorder(
        mode=args.mode,
        env=args.env,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )
    
    recorder.run()

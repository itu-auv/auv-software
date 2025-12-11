import rospy
import argparse
import subprocess
import signal
import sys
import os
import threading
import time
import json
from std_srvs.srv import Empty, Trigger
from auv_smach.gate import NavigateThroughGateState
import smach

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\nShutdown requested...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)

class ExpertRecorder:
    def __init__(self, mode, env, task, num_episodes, output_dir, bag_prefix="expert_data"):
        self.mode = mode
        self.env = env
        self.task = task
        self.num_episodes = num_episodes
        self.output_dir = output_dir
        self.bag_prefix = bag_prefix
        self.bag_process = None
        self.current_bag_path = None
        self.episode_start_time = None
        
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

        # Reset Odometry service
        rospy.loginfo("Waiting for reset_odometry service...")
        try:
            rospy.wait_for_service('reset_odometry', timeout=5.0)
            self.reset_odom_proxy = rospy.ServiceProxy('reset_odometry', Trigger)
            rospy.loginfo("Connected to reset_odometry")
        except rospy.ROSException:
             rospy.logwarn("Service reset_odometry not available.")
             self.reset_odom_proxy = None

        # Clear object transforms service (for resetting TFs)
        rospy.loginfo("Waiting for clear_object_transforms service...")
        try:
            rospy.wait_for_service('clear_object_transforms', timeout=5.0)
            self.clear_transforms_proxy = rospy.ServiceProxy('clear_object_transforms', Trigger)
            rospy.loginfo("Connected to clear_object_transforms")
        except rospy.ROSException:
             rospy.logwarn("Service clear_object_transforms not available. TFs might not be reset.")
             self.clear_transforms_proxy = None

    def get_topics_to_record(self):
        """Get topics based on environment and task."""
        topics = [
            "/taluy/cmd_vel",
            "/taluy/cameras/cam_front/image_raw/compressed",
            "/taluy/odometry",
            "/tf",
            "/tf_static"
        ]
        
        # Add sim-specific topics
        if self.env == 'sim':
            topics.append("/pool_camera/image_raw/compressed")
        
        return topics

    def start_bag_recording(self, episode_idx=None):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if episode_idx is not None:
            bag_name = f"{self.bag_prefix}_{self.task}_ep{episode_idx}_{timestamp}.bag"
        else:
            bag_name = f"{self.bag_prefix}_{self.task}_{timestamp}.bag"
            
        self.current_bag_path = os.path.join(self.output_dir, bag_name)
        self.episode_start_time = time.time()
        
        topics = self.get_topics_to_record()
        cmd = ["rosbag", "record", "-O", self.current_bag_path] + topics
        
        rospy.loginfo(f"Starting recording: {self.current_bag_path}")
        self.bag_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop_bag_recording(self, outcome=None, episode_idx=None):
        if self.bag_process:
            rospy.loginfo("Stopping recording...")
            self.bag_process.send_signal(signal.SIGINT)
            self.bag_process.wait()
            self.bag_process = None
            
            # Save metadata
            if self.current_bag_path and outcome is not None:
                self.save_metadata(outcome, episode_idx)

    def save_metadata(self, outcome, episode_idx=None):
        """Save episode metadata as JSON alongside the bag file."""
        if not self.current_bag_path:
            return
            
        metadata_path = self.current_bag_path.replace('.bag', '_metadata.json')
        
        duration = time.time() - self.episode_start_time if self.episode_start_time else 0
        
        # Determine success/failure
        success = outcome == 'succeeded'
        
        metadata = {
            "bag_file": os.path.basename(self.current_bag_path),
            "task": self.task,
            "mode": self.mode,
            "env": self.env,
            "episode": episode_idx,
            "outcome": outcome,
            "success": success,
            "duration_seconds": round(duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "topics_recorded": self.get_topics_to_record(),
            "ros_params": {
                "gate_depth": rospy.get_param("~gate_depth", -1.35),
                "gate_search_depth": rospy.get_param("~gate_search_depth", -0.7),
                "gate_exit_angle": rospy.get_param("~gate_exit_angle", 0.0),
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        label = "SUCCESS" if success else "FAILURE"
        rospy.loginfo(f"Metadata saved: {metadata_path} [{label}]")

    def reset_sim(self):
        rospy.loginfo("Resetting state...")
        
        # 0. Reset Odometry
        if self.reset_odom_proxy:
            try:
                self.reset_odom_proxy()
                rospy.loginfo("Odometry reset.")
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to reset odometry: {e}")

        # 1. Clear object transforms (TFs)
        if self.clear_transforms_proxy:
            try:
                self.clear_transforms_proxy()
                rospy.loginfo("Object transforms cleared.")
            except rospy.ServiceException as e:
                rospy.logerr(f"Failed to clear transforms: {e}")

        # 2. Reset Gazebo world (only in sim)
        if self.env == 'sim':
            rospy.loginfo("Resetting simulation world...")
            try:
                self.reset_world_proxy()
                rospy.sleep(1.0)
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")

    def run_autonomous_episode(self):
        if self.task == 'gate':
            return self.run_gate_task()
        else:
            rospy.logerr(f"Unknown task: {self.task}")
            return 'aborted'

    def run_gate_task(self):
        """Run the gate navigation task."""
        # Ensure 'roll' and 'yaw' params are False (0) for this task as requested
        rospy.set_param("~roll", False)
        rospy.set_param("~yaw", False)
        
        sm = smach.StateMachine(outcomes=['succeeded', 'preempted', 'aborted'])
        
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
        success_count = 0
        failure_count = 0
        
        for i in range(self.num_episodes):
            if shutdown_requested or rospy.is_shutdown():
                break
                
            rospy.loginfo(f"--- Starting Episode {i+1}/{self.num_episodes} [{self.task}] ---")
            
            # Reset before starting new episode
            self.reset_sim()
            
            self.start_bag_recording(episode_idx=i+1)
            
            outcome = self.run_autonomous_episode()
            
            if outcome == 'succeeded':
                success_count += 1
            else:
                failure_count += 1
                
            rospy.loginfo(f"Episode {i+1} finished: {outcome} (Success: {success_count}, Failure: {failure_count})")
            
            self.stop_bag_recording(outcome=outcome, episode_idx=i+1)
            
            if outcome == 'preempted':
                break
        
        rospy.loginfo(f"=== Recording Complete ===")
        rospy.loginfo(f"Total: {success_count + failure_count}, Success: {success_count}, Failure: {failure_count}")

    def run_human_loop(self):
        rospy.loginfo(f"--- Human Expert Mode [{self.task}] ---")
        if self.env == 'sim':
            rospy.loginfo("Press 'r' + Enter to reset simulation. Press Ctrl+C to exit.")
        else:
            rospy.loginfo("Recording... Press Ctrl+C to stop.")

        self.start_bag_recording()

        def input_listener():
            while not shutdown_requested and not rospy.is_shutdown():
                try:
                    user_input = input()
                    if user_input.strip() == 'r':
                        # Stop first
                        self.stop_bag_recording(outcome='reset', episode_idx=None)
                        # Then reset sim and TFs
                        self.reset_sim()
                        # Then start new bag
                        self.start_bag_recording()
                except EOFError:
                    break

        if self.env == 'sim':
            t = threading.Thread(target=input_listener)
            t.daemon = True
            t.start()

        rate = rospy.Rate(10)
        while not shutdown_requested and not rospy.is_shutdown():
            rate.sleep()
            
        self.stop_bag_recording(outcome='manual_stop', episode_idx=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record expert data for AUV")
    parser.add_argument("--mode", default="autonomous", choices=["autonomous", "human"], help="Recording mode")
    parser.add_argument("--env", default="sim", choices=["sim", "real"], help="Environment type")
    parser.add_argument("--task", default="gate", choices=["gate"], help="Task to record")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes (autonomous mode)")
    parser.add_argument("--output_dir", default=os.path.join(os.environ.get("HOME"), "bags/expert_data"), help="Output directory for bags")
    
    args = parser.parse_args(rospy.myargv()[1:])
    
    recorder = ExpertRecorder(
        mode=args.mode,
        env=args.env,
        task=args.task,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir
    )
    
    recorder.run()

#!/usr/bin/env python3
"""
RL Agent Node - Deploy trained model for real-time navigation
Loads a trained PPO model and runs inference in real-time
"""

import rospy
import numpy as np
from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import PoseStamped
from auv_rl_navigation.environments.auv_nav_env import AUVNavEnv
from auv_rl_navigation.agents.ppo_agent import AUVPPOAgent
import os


class RLAgentNode:
    """
    ROS node for running trained RL agent in deployment mode.
    """

    def __init__(self):
        """Initialize the RL agent node."""
        rospy.init_node("rl_agent_node", anonymous=False)

        rospy.loginfo("=" * 60)
        rospy.loginfo("RL Agent Node - Deployment Mode")
        rospy.loginfo("=" * 60)

        # Parameters
        self.model_path = rospy.get_param("~model_path", "./models/ppo_auv_best.zip")
        self.goal_frame = rospy.get_param("~goal_frame", "gate")
        self.rate_hz = rospy.get_param("~rate", 10)  # Control frequency
        self.deterministic = rospy.get_param("~deterministic", True)  # No exploration
        self.visualize = rospy.get_param("~visualize", True)
        self.max_episode_steps = rospy.get_param("~max_episode_steps", 1000)
        self.goal_tolerance = rospy.get_param("~goal_tolerance", 1.0)

        # Object frames
        object_frames_str = rospy.get_param(
            "~object_frames",
            "gate_shark_link,gate_sawfish_link,red_pipe_link,white_pipe_link,"
            "red_buoy,torpedo_map_link,bin_whole_link,octagon_link",
        )
        self.object_frames = [f.strip() for f in object_frames_str.split(",")]

        # State
        self.enabled = rospy.get_param("~start_enabled", True)
        self.episode_count = 0
        self.step_count = 0
        self.cumulative_reward = 0.0

        rospy.loginfo(f"Configuration:")
        rospy.loginfo(f"  Model: {self.model_path}")
        rospy.loginfo(f"  Goal frame: {self.goal_frame}")
        rospy.loginfo(f"  Rate: {self.rate_hz} Hz")
        rospy.loginfo(f"  Deterministic: {self.deterministic}")
        rospy.loginfo(f"  Enabled at start: {self.enabled}")

        # Create environment (for observation processing)
        rospy.loginfo("\nCreating environment...")
        self.env = AUVNavEnv(
            max_episode_steps=self.max_episode_steps,
            goal_tolerance=self.goal_tolerance,
            object_frames=self.object_frames,
            goal_frame=self.goal_frame,
            visualize=self.visualize,
        )
        rospy.loginfo("Environment created!")

        # Load trained model
        rospy.loginfo(f"\nLoading model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            rospy.logerr(f"Model file not found: {self.model_path}")
            rospy.logerr("Please train a model first or specify correct path!")
            rospy.signal_shutdown("Model not found")
            return

        self.agent = AUVPPOAgent(self.env)
        self.agent.load(self.model_path)
        rospy.loginfo("Model loaded successfully!")

        # Publishers for monitoring
        self.reward_pub = rospy.Publisher("~reward", Float32, queue_size=1)
        self.distance_pub = rospy.Publisher("~distance_to_goal", Float32, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1)
        self.action_pub = rospy.Publisher("~current_action", PoseStamped, queue_size=1)

        # Subscribers for control
        self.enable_sub = rospy.Subscriber("~enable", Bool, self._enable_callback)
        self.reset_sub = rospy.Subscriber("~reset", Bool, self._reset_callback)

        # Control rate
        self.rate = rospy.Rate(self.rate_hz)

        # Initialize episode
        self.current_obs = self.env.reset()
        self.episode_count = 1

        rospy.loginfo("\n" + "=" * 60)
        rospy.loginfo("RL Agent ready! Starting navigation...")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Control topics:")
        rospy.loginfo("  - ~/enable (std_msgs/Bool) - Enable/disable agent")
        rospy.loginfo("  - ~/reset (std_msgs/Bool) - Reset episode")
        rospy.loginfo("Monitoring topics:")
        rospy.loginfo("  - ~/reward (std_msgs/Float32)")
        rospy.loginfo("  - ~/distance_to_goal (std_msgs/Float32)")
        rospy.loginfo("  - ~/status (std_msgs/String)")
        rospy.loginfo("=" * 60 + "\n")

    def _enable_callback(self, msg):
        """Enable/disable the agent."""
        self.enabled = msg.data
        if self.enabled:
            rospy.loginfo("🟢 RL Agent ENABLED")
            self.current_obs = self.env.reset()
        else:
            rospy.logwarn("🔴 RL Agent DISABLED")
            # Stop the robot
            self.env._publish_action(np.zeros(4))

    def _reset_callback(self, msg):
        """Reset the episode."""
        if msg.data:
            rospy.loginfo("🔄 Resetting episode...")
            self.current_obs = self.env.reset()
            self.step_count = 0
            self.cumulative_reward = 0.0
            self.episode_count += 1

    def run(self):
        """Main control loop."""

        while not rospy.is_shutdown():
            if self.enabled:
                # Get action from policy
                action, _ = self.agent.predict(
                    self.current_obs, deterministic=self.deterministic
                )

                # Execute action
                obs, reward, done, info = self.env.step(action)
                self.current_obs = obs

                # Update counters
                self.step_count += 1
                self.cumulative_reward += reward

                # Publish monitoring info
                self.reward_pub.publish(Float32(reward))

                if "distance_to_goal" in info:
                    self.distance_pub.publish(Float32(info["distance_to_goal"]))

                # Publish action (for visualization)
                action_msg = PoseStamped()
                action_msg.header.stamp = rospy.Time.now()
                action_msg.header.frame_id = "action"
                action_msg.pose.position.x = action[0]  # surge
                action_msg.pose.position.y = action[1]  # sway
                action_msg.pose.position.z = action[2]  # heave
                action_msg.pose.orientation.z = action[3]  # yaw_rate
                self.action_pub.publish(action_msg)

                # Log progress
                if self.step_count % 50 == 0:
                    rospy.loginfo(
                        f"Episode {self.episode_count}, Step {self.step_count}: "
                        f"reward={self.cumulative_reward:.2f}, "
                        f"distance={info.get('distance_to_goal', 'N/A')}"
                    )

                # Publish status
                status = (
                    f"Episode {self.episode_count} | Step {self.step_count} | "
                    f"Reward {self.cumulative_reward:.1f}"
                )
                self.status_pub.publish(String(status))

                # Check episode termination
                if done:
                    reason = info.get("termination_reason", "unknown")
                    rospy.loginfo("\n" + "-" * 60)
                    rospy.loginfo(f"Episode {self.episode_count} finished!")
                    rospy.loginfo(f"  Reason: {reason}")
                    rospy.loginfo(f"  Steps: {self.step_count}")
                    rospy.loginfo(f"  Cumulative reward: {self.cumulative_reward:.2f}")
                    rospy.loginfo("-" * 60 + "\n")

                    # Reset for next episode
                    self.current_obs = self.env.reset()
                    self.step_count = 0
                    self.cumulative_reward = 0.0
                    self.episode_count += 1

                    rospy.sleep(2.0)  # Pause between episodes

            else:
                # Agent disabled, just wait
                self.status_pub.publish(String("DISABLED"))

            self.rate.sleep()

    def shutdown(self):
        """Clean shutdown."""
        rospy.loginfo("\nShutting down RL agent...")
        self.env.close()
        rospy.loginfo("RL agent stopped.")


def main():
    """Main function."""
    try:
        node = RLAgentNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("RL agent node terminated.")
    except Exception as e:
        rospy.logerr(f"RL agent node failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if "node" in locals():
            node.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training Progress Visualization
Real-time plotting of training metrics using matplotlib
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from std_msgs.msg import Float32, String
from collections import deque
import threading


class TrainingMonitor:
    """
    Real-time monitor for RL training progress.
    Displays live plots of rewards, episode lengths, and success rates.
    """

    def __init__(self, window_size=100):
        """
        Initialize the training monitor.

        Args:
            window_size: Number of episodes to display in rolling window
        """
        rospy.init_node("training_monitor", anonymous=True)

        self.window_size = window_size

        # Data buffers
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.distances = deque(maxlen=1000)  # More samples for distance
        self.success_history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        self.episode_count = 0

        self.start_time = rospy.Time.now()

        # Thread lock for data safety
        self.lock = threading.Lock()

        # Subscribe to training metrics
        rospy.Subscriber("/auv_rl_trainer/reward", Float32, self._reward_callback)
        rospy.Subscriber(
            "/auv_rl_trainer/distance_to_goal", Float32, self._distance_callback
        )
        rospy.Subscriber("/auv_rl_trainer/status", String, self._status_callback)

        # Also support deployment node topics
        rospy.Subscriber("/rl_agent_node/reward", Float32, self._reward_callback)
        rospy.Subscriber(
            "/rl_agent_node/distance_to_goal", Float32, self._distance_callback
        )
        rospy.Subscriber("/rl_agent_node/status", String, self._status_callback)

        rospy.loginfo("Training Monitor started!")
        rospy.loginfo("Subscribing to training metrics...")
        rospy.loginfo("Close the plot window to exit.")

    def _reward_callback(self, msg):
        """Callback for reward messages."""
        with self.lock:
            self.current_episode_reward += msg.data
            self.current_episode_steps += 1

    def _distance_callback(self, msg):
        """Callback for distance messages."""
        with self.lock:
            self.distances.append(msg.data)

    def _status_callback(self, msg):
        """Callback for status messages - detect episode end."""
        status = msg.data

        # Check if episode ended (simple heuristic)
        if "Episode" in status:
            with self.lock:
                if self.current_episode_steps > 0:
                    # Save episode data
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episode_lengths.append(self.current_episode_steps)

                    # Determine success (reward > threshold)
                    success = self.current_episode_reward > 50.0
                    self.success_history.append(1 if success else 0)

                    elapsed = (rospy.Time.now() - self.start_time).to_sec()
                    self.timestamps.append(elapsed / 60.0)  # minutes

                    self.episode_count += 1

                    # Reset for next episode
                    self.current_episode_reward = 0.0
                    self.current_episode_steps = 0

    def create_plots(self):
        """Create the matplotlib figure and subplots."""
        plt.style.use("seaborn-v0_8-darkgrid")
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle(
            "AUV RL Navigation - Training Progress", fontsize=16, fontweight="bold"
        )

        # Subplot 1: Episode Rewards
        self.ax_reward = self.axes[0, 0]
        self.ax_reward.set_title("Episode Rewards")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Total Reward")
        self.ax_reward.grid(True, alpha=0.3)

        # Subplot 2: Episode Lengths
        self.ax_length = self.axes[0, 1]
        self.ax_length.set_title("Episode Lengths")
        self.ax_length.set_xlabel("Episode")
        self.ax_length.set_ylabel("Steps")
        self.ax_length.grid(True, alpha=0.3)

        # Subplot 3: Success Rate (rolling average)
        self.ax_success = self.axes[1, 0]
        self.ax_success.set_title("Success Rate (Rolling Average)")
        self.ax_success.set_xlabel("Episode")
        self.ax_success.set_ylabel("Success Rate")
        self.ax_success.set_ylim([0, 1])
        self.ax_success.grid(True, alpha=0.3)

        # Subplot 4: Distance to Goal
        self.ax_distance = self.axes[1, 1]
        self.ax_distance.set_title("Distance to Goal (Recent)")
        self.ax_distance.set_xlabel("Step")
        self.ax_distance.set_ylabel("Distance (m)")
        self.ax_distance.grid(True, alpha=0.3)

        plt.tight_layout()

    def update_plots(self, frame):
        """Update plots with new data."""
        with self.lock:
            # Skip if no data
            if len(self.episode_rewards) == 0:
                return

            # Clear all axes
            for ax in self.axes.flat:
                ax.clear()

            episodes = list(range(len(self.episode_rewards)))

            # Plot 1: Episode Rewards
            self.ax_reward.plot(
                episodes, list(self.episode_rewards), "b-", alpha=0.6, linewidth=1
            )
            if len(self.episode_rewards) > 10:
                # Moving average
                window = min(10, len(self.episode_rewards))
                moving_avg = np.convolve(
                    self.episode_rewards, np.ones(window) / window, mode="valid"
                )
                self.ax_reward.plot(
                    range(window - 1, len(self.episode_rewards)),
                    moving_avg,
                    "r-",
                    linewidth=2,
                    label="Moving Avg (10)",
                )
                self.ax_reward.legend()
            self.ax_reward.set_title("Episode Rewards")
            self.ax_reward.set_xlabel("Episode")
            self.ax_reward.set_ylabel("Total Reward")
            self.ax_reward.grid(True, alpha=0.3)

            # Plot 2: Episode Lengths
            self.ax_length.plot(
                episodes, list(self.episode_lengths), "g-", alpha=0.6, linewidth=1
            )
            if len(self.episode_lengths) > 10:
                window = min(10, len(self.episode_lengths))
                moving_avg = np.convolve(
                    self.episode_lengths, np.ones(window) / window, mode="valid"
                )
                self.ax_length.plot(
                    range(window - 1, len(self.episode_lengths)),
                    moving_avg,
                    "darkgreen",
                    linewidth=2,
                    label="Moving Avg (10)",
                )
                self.ax_length.legend()
            self.ax_length.set_title("Episode Lengths")
            self.ax_length.set_xlabel("Episode")
            self.ax_length.set_ylabel("Steps")
            self.ax_length.grid(True, alpha=0.3)

            # Plot 3: Success Rate
            if len(self.success_history) > 0:
                success_rate = []
                window = min(20, len(self.success_history))
                for i in range(len(self.success_history)):
                    start_idx = max(0, i - window + 1)
                    rate = np.mean(list(self.success_history)[start_idx : i + 1])
                    success_rate.append(rate)

                self.ax_success.plot(episodes, success_rate, "purple", linewidth=2)
                self.ax_success.fill_between(
                    episodes, 0, success_rate, alpha=0.3, color="purple"
                )

                # Show current success rate
                current_rate = success_rate[-1] if success_rate else 0
                self.ax_success.text(
                    0.02,
                    0.98,
                    f"Current: {current_rate:.1%}",
                    transform=self.ax_success.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            self.ax_success.set_title("Success Rate (Rolling Average)")
            self.ax_success.set_xlabel("Episode")
            self.ax_success.set_ylabel("Success Rate")
            self.ax_success.set_ylim([0, 1])
            self.ax_success.grid(True, alpha=0.3)

            # Plot 4: Distance to Goal (recent history)
            if len(self.distances) > 0:
                recent_distances = list(self.distances)[-500:]  # Last 500 steps
                self.ax_distance.plot(
                    recent_distances, "orange", alpha=0.7, linewidth=1
                )
                self.ax_distance.axhline(
                    y=1.0, color="r", linestyle="--", label="Goal Tolerance", alpha=0.7
                )
                self.ax_distance.legend()

            self.ax_distance.set_title("Distance to Goal (Recent)")
            self.ax_distance.set_xlabel("Step")
            self.ax_distance.set_ylabel("Distance (m)")
            self.ax_distance.grid(True, alpha=0.3)

            # Update main title with episode count
            self.fig.suptitle(
                f"AUV RL Navigation - Training Progress (Episode {self.episode_count})",
                fontsize=16,
                fontweight="bold",
            )

    def run(self):
        """Run the monitor with live plotting."""
        self.create_plots()

        # Animate with 1 second update interval
        ani = FuncAnimation(
            self.fig, self.update_plots, interval=1000, cache_frame_data=False
        )

        plt.show()

        rospy.loginfo("Monitor window closed.")


def main():
    """Main function."""
    try:
        monitor = TrainingMonitor(window_size=100)
        monitor.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nMonitor interrupted.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Live Action Monitor
Real-time visualization of agent's actions and observations
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Float32
import threading


class LiveActionMonitor:
    """
    Real-time monitor for agent's actions and state.
    """

    def __init__(self):
        """Initialize the live action monitor."""
        rospy.init_node("live_action_monitor", anonymous=True)

        # Data buffers
        self.actions = {"surge": [], "sway": [], "heave": [], "yaw_rate": []}
        self.velocities = {"vx": [], "vy": [], "vz": [], "yaw": []}
        self.distance = []
        self.reward = []
        self.max_history = 100

        self.lock = threading.Lock()

        # Subscribe to topics
        rospy.Subscriber(
            "/rl_agent_node/current_action", PoseStamped, self._action_callback
        )
        rospy.Subscriber("/taluy/cmd_vel", Twist, self._cmd_vel_callback)
        rospy.Subscriber(
            "/rl_agent_node/distance_to_goal", Float32, self._distance_callback
        )
        rospy.Subscriber("/rl_agent_node/reward", Float32, self._reward_callback)

        rospy.loginfo("Live Action Monitor started!")

    def _action_callback(self, msg):
        """Callback for action messages."""
        with self.lock:
            self.actions["surge"].append(msg.pose.position.x)
            self.actions["sway"].append(msg.pose.position.y)
            self.actions["heave"].append(msg.pose.position.z)
            self.actions["yaw_rate"].append(msg.pose.orientation.z)

            # Keep only recent history
            for key in self.actions:
                if len(self.actions[key]) > self.max_history:
                    self.actions[key].pop(0)

    def _cmd_vel_callback(self, msg):
        """Callback for cmd_vel messages."""
        with self.lock:
            self.velocities["vx"].append(msg.linear.x)
            self.velocities["vy"].append(msg.linear.y)
            self.velocities["vz"].append(msg.linear.z)
            self.velocities["yaw"].append(msg.angular.z)

            for key in self.velocities:
                if len(self.velocities[key]) > self.max_history:
                    self.velocities[key].pop(0)

    def _distance_callback(self, msg):
        """Callback for distance messages."""
        with self.lock:
            self.distance.append(msg.data)
            if len(self.distance) > self.max_history:
                self.distance.pop(0)

    def _reward_callback(self, msg):
        """Callback for reward messages."""
        with self.lock:
            self.reward.append(msg.data)
            if len(self.reward) > self.max_history:
                self.reward.pop(0)

    def create_plots(self):
        """Create the matplotlib figure."""
        plt.style.use("seaborn-v0_8-darkgrid")
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle(
            "AUV RL Agent - Live Action Monitor", fontsize=16, fontweight="bold"
        )

        # Subplot 1: Actions
        self.ax_action = self.axes[0]
        self.ax_action.set_title("Agent Actions (RL Output)")
        self.ax_action.set_ylabel("Action Value")
        self.ax_action.grid(True, alpha=0.3)
        self.ax_action.legend(["Surge", "Sway", "Heave", "Yaw Rate"], loc="upper right")

        # Subplot 2: Distance & Reward
        self.ax_perf = self.axes[1]
        self.ax_perf.set_title("Performance Metrics")
        self.ax_perf.set_ylabel("Value")
        self.ax_perf.grid(True, alpha=0.3)

        # Subplot 3: Velocity Commands
        self.ax_vel = self.axes[2]
        self.ax_vel.set_title("Velocity Commands (to /cmd_vel)")
        self.ax_vel.set_xlabel("Time Steps (recent 100)")
        self.ax_vel.set_ylabel("Velocity")
        self.ax_vel.grid(True, alpha=0.3)
        self.ax_vel.legend(["Vx", "Vy", "Vz", "Yaw"], loc="upper right")

        plt.tight_layout()

    def update_plots(self, frame):
        """Update plots with new data."""
        with self.lock:
            # Clear axes
            for ax in self.axes:
                ax.clear()

            steps = range(self.max_history)

            # Plot 1: Actions
            if len(self.actions["surge"]) > 0:
                n = len(self.actions["surge"])
                x = range(n)
                self.ax_action.plot(
                    x, self.actions["surge"], "b-", label="Surge", linewidth=2
                )
                self.ax_action.plot(
                    x, self.actions["sway"], "g-", label="Sway", linewidth=2
                )
                self.ax_action.plot(
                    x, self.actions["heave"], "r-", label="Heave", linewidth=2
                )
                self.ax_action.plot(
                    x, self.actions["yaw_rate"], "m-", label="Yaw Rate", linewidth=2
                )
                self.ax_action.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                self.ax_action.set_ylim([-1.2, 1.2])

            self.ax_action.set_title("Agent Actions (RL Output)")
            self.ax_action.set_ylabel("Action Value [-1, 1]")
            self.ax_action.grid(True, alpha=0.3)
            self.ax_action.legend(loc="upper right")

            # Plot 2: Distance & Reward
            if len(self.distance) > 0 or len(self.reward) > 0:
                ax2 = self.ax_perf.twinx()

                if len(self.distance) > 0:
                    self.ax_perf.plot(
                        range(len(self.distance)),
                        self.distance,
                        "orange",
                        linewidth=2,
                        label="Distance to Goal",
                    )
                    self.ax_perf.axhline(
                        y=1.0,
                        color="r",
                        linestyle="--",
                        alpha=0.5,
                        label="Goal Tolerance",
                    )
                    self.ax_perf.set_ylabel("Distance (m)", color="orange")
                    self.ax_perf.tick_params(axis="y", labelcolor="orange")

                if len(self.reward) > 0:
                    ax2.plot(
                        range(len(self.reward)),
                        self.reward,
                        "purple",
                        linewidth=2,
                        label="Reward",
                        alpha=0.7,
                    )
                    ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
                    ax2.set_ylabel("Reward", color="purple")
                    ax2.tick_params(axis="y", labelcolor="purple")

                # Combine legends
                lines1, labels1 = self.ax_perf.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                self.ax_perf.legend(
                    lines1 + lines2, labels1 + labels2, loc="upper right"
                )

            self.ax_perf.set_title("Performance Metrics")
            self.ax_perf.grid(True, alpha=0.3)

            # Plot 3: Velocities
            if len(self.velocities["vx"]) > 0:
                n = len(self.velocities["vx"])
                x = range(n)
                self.ax_vel.plot(
                    x, self.velocities["vx"], "b-", label="Vx (surge)", linewidth=2
                )
                self.ax_vel.plot(
                    x, self.velocities["vy"], "g-", label="Vy (sway)", linewidth=2
                )
                self.ax_vel.plot(
                    x, self.velocities["vz"], "r-", label="Vz (heave)", linewidth=2
                )
                self.ax_vel.plot(
                    x, self.velocities["yaw"], "m-", label="Yaw Rate", linewidth=2
                )
                self.ax_vel.axhline(y=0, color="k", linestyle="--", alpha=0.3)

            self.ax_vel.set_title("Velocity Commands (to /cmd_vel)")
            self.ax_vel.set_xlabel("Time Steps (recent 100)")
            self.ax_vel.set_ylabel("Velocity (m/s, rad/s)")
            self.ax_vel.grid(True, alpha=0.3)
            self.ax_vel.legend(loc="upper right")

    def run(self):
        """Run the monitor with live plotting."""
        self.create_plots()

        # Animate with 200ms update interval
        ani = FuncAnimation(
            self.fig, self.update_plots, interval=200, cache_frame_data=False
        )

        plt.show()

        rospy.loginfo("Monitor window closed.")


def main():
    """Main function."""
    try:
        monitor = LiveActionMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nMonitor interrupted.")


if __name__ == "__main__":
    main()

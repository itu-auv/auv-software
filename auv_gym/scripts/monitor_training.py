#!/usr/bin/env python3
import rospy
import json
import curses
import sys
from std_msgs.msg import String


class TrainingMonitor:
    def __init__(self):
        rospy.init_node("training_monitor", anonymous=True)

        # Get namespace from param or arg, default to 'taluy'
        self.ns = rospy.get_param("~ns", "taluy")

        topic_name = f"/{self.ns}/training_status"
        rospy.loginfo(f"Subscribing to {topic_name}")
        self.sub = rospy.Subscriber(topic_name, String, self.callback)
        self.data = None
        self.stdscr = None

    def callback(self, msg):
        try:
            self.data = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def draw_dashboard(self):
        if not self.stdscr:
            return

        if not self.data:
            self.stdscr.clear()
            self.stdscr.addstr(
                0, 0, f"Waiting for training data on /{self.ns}/training_status..."
            )
            self.stdscr.refresh()
            return

        try:
            self.stdscr.clear()
            self.stdscr.border()

            # Title
            title = " AUV RL Training Monitor "
            self.stdscr.addstr(0, 2, title, curses.A_BOLD)

            # Episode Info
            self.stdscr.addstr(2, 2, "EPISODE INFO", curses.A_UNDERLINE)
            self.stdscr.addstr(3, 4, f"Episode:     {self.data.get('episode', 0)}")
            self.stdscr.addstr(4, 4, f"Step:        {self.data.get('step', 0)}")
            self.stdscr.addstr(5, 4, f"Total Steps: {self.data.get('total_steps', 0)}")

            # Performance
            self.stdscr.addstr(7, 2, "PERFORMANCE", curses.A_UNDERLINE)
            self.stdscr.addstr(
                8, 4, f"Current Reward:     {self.data.get('current_reward', 0.0):.2f}"
            )
            self.stdscr.addstr(
                9, 4, f"Mean Reward (100):  {self.data.get('mean_reward_100', 0.0):.2f}"
            )
            self.stdscr.addstr(
                10,
                4,
                f"Success Rate (100): {self.data.get('success_rate_100', 0.0)*100:.1f}%",
            )
            self.stdscr.addstr(
                11, 4, f"Successes:          {self.data.get('success_count', 0)}"
            )
            self.stdscr.addstr(
                12, 4, f"Failures:           {self.data.get('fail_count', 0)}"
            )

            # State
            self.stdscr.addstr(2, 40, "STATE", curses.A_UNDERLINE)
            self.stdscr.addstr(
                3,
                42,
                f"Dist to Target: {self.data.get('distance_to_target', 0.0):.2f} m",
            )
            self.stdscr.addstr(
                4, 42, f"Heading Error:  {self.data.get('heading_error', 0.0):.2f} rad"
            )
            self.stdscr.addstr(
                5, 42, f"Velocity:       {self.data.get('velocity', 0.0):.2f} m/s"
            )
            self.stdscr.addstr(
                6, 42, f"Action Mag:     {self.data.get('action_magnitude', 0.0):.2f}"
            )

            # System
            self.stdscr.addstr(8, 40, "SYSTEM", curses.A_UNDERLINE)
            self.stdscr.addstr(
                9, 42, f"RTF (Sim/Real): {self.data.get('rtf', 0.0):.2f}x"
            )

            self.stdscr.refresh()
        except curses.error:
            pass

    def run(self):
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.curs_set(0)

        try:
            rate = rospy.Rate(10)  # 10Hz refresh
            while not rospy.is_shutdown():
                self.draw_dashboard()
                rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            if self.stdscr:
                curses.nocbreak()
                self.stdscr.keypad(False)
                curses.echo()
                curses.endwin()


if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run()

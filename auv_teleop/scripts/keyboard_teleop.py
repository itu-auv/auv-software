#!/usr/bin/env python3
"""
Terminal-based keyboard teleoperation for AUV.
Works similarly to joystick_node: continuously publishes velocity commands
while driving mode is enabled.

Key release is simulated via timeout - if a key isn't re-pressed within
KEY_TIMEOUT_SEC, it's considered released.
"""
from __future__ import annotations

import sys
import tty
import termios
import select
import time
from typing import Optional, Dict

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

KEY_TIMEOUT_SEC = 0.6  # Must be > keyboard repeat delay (~500ms)


class KeyboardTeleop:
    """Non-blocking keyboard teleop with terminal UI."""

    HELP_TEXT = """
╔═══════════════════════════════════════════════════════════╗
║                   AUV KEYBOARD TELEOP                     ║
╠═══════════════════════════════════════════════════════════╣
║  Movement (HOLD keys):                                    ║
║    W / S  →  Forward / Backward  (X axis)                 ║
║    A / D  →  Left / Right        (Y axis)                 ║
║    X      →  Ascend              (Z axis +)               ║
║    Z      →  Descend             (Z axis -)               ║
║                                                           ║
║  Rotation (HOLD keys):                                    ║
║    Q      →  Yaw Left                                     ║
║    E      →  Yaw Right                                    ║
║                                                           ║
║  Control:                                                 ║
║  Ctrl+SPACE  →  Enable driving mode                       ║
║  SPACE       →  Disable driving mode                      ║
║  Ctrl+C      →  Exit                                      ║
║                                                           ║
║  When no key is held, velocity = 0                        ║
╚═══════════════════════════════════════════════════════════╝
"""

    MOVEMENT_KEYS = {"w", "a", "s", "d", "z", "x", "q", "e"}

    def __init__(self) -> None:
        rospy.init_node("keyboard_teleop_node", anonymous=True)

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=10)

        self.linear_speed = rospy.get_param("~linear_speed", 0.3)
        self.angular_speed = rospy.get_param("~angular_speed", 0.3)
        self.rate_hz = rospy.get_param("~rate", 50)

        self.driving_enabled = False
        self.current_twist = Twist()
        # key -> last_press_time
        self.key_timestamps: dict[str, float] = {}

        self.old_settings = termios.tcgetattr(sys.stdin)
        self.last_print_time = 0.0

    def _key_available(self) -> bool:
        return select.select([sys.stdin], [], [], 0)[0] != []

    def _read_key(self) -> str | None:
        if not self._key_available():
            return None

        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence
            if self._key_available():
                sys.stdin.read(2)  # consume bracket + code
            return None
        return ch.lower() if ch.isalpha() else ch

    def _expire_old_keys(self) -> None:
        """Remove keys that haven't been pressed recently (simulates key release)."""
        now = time.time()
        expired = [k for k, t in self.key_timestamps.items() if now - t > KEY_TIMEOUT_SEC]
        for k in expired:
            del self.key_timestamps[k]

    def _compute_twist(self) -> Twist:
        """Compute twist based on currently active keys."""
        twist = Twist()
        keys = set(self.key_timestamps.keys())

        if "w" in keys:
            twist.linear.x += self.linear_speed
        if "s" in keys:
            twist.linear.x -= self.linear_speed

        if "a" in keys:
            twist.linear.y += self.linear_speed
        if "d" in keys:
            twist.linear.y -= self.linear_speed

        if "x" in keys:
            twist.linear.z += self.linear_speed
        if "z" in keys:
            twist.linear.z -= self.linear_speed

        if "q" in keys:
            twist.angular.z += self.angular_speed
        if "e" in keys:
            twist.angular.z -= self.angular_speed

        return twist

    def _print_status(self, force: bool = False) -> None:
        """Clear screen and print current status. Throttled to avoid flicker."""
        now = time.time()
        if not force and now - self.last_print_time < 0.1:
            return
        self.last_print_time = now

        # ANSI: move cursor to home + clear screen
        sys.stdout.write("\033[H\033[J")
        print(self.HELP_TEXT)

        if self.driving_enabled:
            status = "\033[1;32m● SÜRÜŞ MODU: AKTİF\033[0m"  # green bold
        else:
            status = "\033[1;31m○ SÜRÜŞ MODU: PASİF\033[0m"  # red bold

        print(f"  {status}")
        print(f"\n  Linear Speed:  {self.linear_speed:.2f} m/s")
        print(f"  Angular Speed: {self.angular_speed:.2f} rad/s")

        t = self.current_twist
        print(f"\n  Anlık Hız:")
        print(f"    X: {t.linear.x:+.2f}  Y: {t.linear.y:+.2f}  Z: {t.linear.z:+.2f}")
        print(f"    Yaw: {t.angular.z:+.2f}")

        active = ", ".join(sorted(self.key_timestamps.keys())) if self.key_timestamps else "(yok)"
        print(f"\n  Aktif Tuşlar: {active}")

        sys.stdout.flush()

    def run(self) -> None:
        rate = rospy.Rate(self.rate_hz)

        try:
            tty.setcbreak(sys.stdin.fileno())
            self._print_status(force=True)

            while not rospy.is_shutdown():
                key = self._read_key()

                if key:
                    if key == "\x00":  # Ctrl+Space - enable
                        self.driving_enabled = True
                    elif key == " ":  # Space - disable
                        self.driving_enabled = False
                        self.key_timestamps.clear()
                        self.current_twist = Twist()

                    elif key in self.MOVEMENT_KEYS and self.driving_enabled:
                        self.key_timestamps[key] = time.time()

                # Expire old keys and update twist
                self._expire_old_keys()
                self.current_twist = self._compute_twist() if self.driving_enabled else Twist()

                # Publish at fixed rate
                if self.driving_enabled:
                    self.enable_pub.publish(Bool(True))
                    self.cmd_vel_pub.publish(self.current_twist)
                else:
                    self.enable_pub.publish(Bool(False))

                self._print_status()
                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            # Send zero velocity on exit
            self.cmd_vel_pub.publish(Twist())
            self.enable_pub.publish(Bool(False))
            print("\nKeyboard teleop terminated.")


if __name__ == "__main__":
    try:
        node = KeyboardTeleop()
        node.run()
    except rospy.ROSInterruptException:
        pass

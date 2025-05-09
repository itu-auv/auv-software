#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)
import subprocess
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import threading
import time


class VehicleControlTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=10)

        self.init_ui()
        self.publishing = False
        self.current_twist = Twist()
        self.enable_thread = None
        self.enable_publishing = False
        self.launch_process = None

    def init_ui(self):
        layout = QGridLayout()

        teleop_group = QGroupBox("Teleoperation")
        teleop_layout = QGridLayout()
        self.teleop_start = QPushButton("Start Teleop")
        self.start_enable = QPushButton("Enable Control")
        self.stop_enable = QPushButton("Disable Control")
        self.xbox_check = QCheckBox("Xbox")
        self.teleop_stop = QPushButton("Stop Teleop")
        teleop_layout.addWidget(self.teleop_start, 0, 0)
        teleop_layout.addWidget(self.start_enable, 1, 0)
        teleop_layout.addWidget(self.stop_enable, 1, 2)
        teleop_layout.addWidget(self.xbox_check, 0, 1)
        teleop_layout.addWidget(self.teleop_stop, 0, 2)
        teleop_group.setLayout(teleop_layout)

        control_group = QGroupBox("Manual Control")
        control_layout = QGridLayout()

        button_size = 60

        self.forward_btn = QPushButton("^")
        self.forward_btn.setFixedSize(button_size, button_size)
        self.left_btn = QPushButton("<")
        self.left_btn.setFixedSize(button_size, button_size)
        self.right_btn = QPushButton(">")
        self.right_btn.setFixedSize(button_size, button_size)
        self.backward_btn = QPushButton("v")
        self.backward_btn.setFixedSize(button_size, button_size)
        self.up_btn = QPushButton("Up")
        self.up_btn.setFixedSize(button_size, button_size)
        self.down_btn = QPushButton("Down")
        self.down_btn.setFixedSize(button_size, button_size)
        self.yaw_left_btn = QPushButton("Left")
        self.yaw_left_btn.setFixedSize(button_size, button_size)
        self.yaw_right_btn = QPushButton("Right")
        self.yaw_right_btn.setFixedSize(button_size, button_size)

        self.connect_publishing_button(
            self.forward_btn, "forward", self.get_linear_spinbox
        )
        self.connect_publishing_button(self.left_btn, "left", self.get_linear_spinbox)
        self.connect_publishing_button(self.right_btn, "right", self.get_linear_spinbox)
        self.connect_publishing_button(
            self.backward_btn, "backward", self.get_linear_spinbox
        )
        self.connect_publishing_button(self.up_btn, "pos_z", self.get_linear_spinbox)
        self.connect_publishing_button(self.down_btn, "neg_z", self.get_linear_spinbox)
        self.connect_publishing_button(
            self.yaw_left_btn, "pos_yaw", self.get_angular_spinbox
        )
        self.connect_publishing_button(
            self.yaw_right_btn, "neg_yaw", self.get_angular_spinbox
        )

        control_layout.addWidget(self.forward_btn, 0, 1)
        control_layout.addWidget(self.left_btn, 1, 0)
        control_layout.addWidget(self.right_btn, 1, 2)
        control_layout.addWidget(self.backward_btn, 2, 1)
        control_layout.addWidget(self.up_btn, 0, 4)
        control_layout.addWidget(self.down_btn, 2, 4)
        control_layout.addWidget(self.yaw_left_btn, 1, 3)
        control_layout.addWidget(self.yaw_right_btn, 1, 5)

        speed_layout = QHBoxLayout()

        linear_label = QLabel("Linear Speed:")
        self.linear_speed_spinbox = QDoubleSpinBox()
        self.linear_speed_spinbox.setRange(0.1, 0.4)
        self.linear_speed_spinbox.setSingleStep(0.1)
        self.linear_speed_spinbox.setValue(0.2)

        angular_label = QLabel("Angular Speed:")
        self.angular_speed_spinbox = QDoubleSpinBox()
        self.angular_speed_spinbox.setRange(0.1, 0.4)
        self.angular_speed_spinbox.setSingleStep(0.1)
        self.angular_speed_spinbox.setValue(0.2)

        speed_layout.addWidget(linear_label)
        speed_layout.addWidget(self.linear_speed_spinbox)
        speed_layout.addWidget(angular_label)
        speed_layout.addWidget(self.angular_speed_spinbox)

        control_vlayout = QVBoxLayout()
        control_vlayout.addLayout(control_layout)
        control_vlayout.addSpacing(10)
        control_vlayout.addLayout(speed_layout)
        control_group.setLayout(control_vlayout)

        layout.addWidget(teleop_group, 0, 0, 1, 2)
        layout.addWidget(control_group, 1, 0, 1, 2)
        self.setLayout(layout)

        self.teleop_start.clicked.connect(self.start_teleop)
        self.teleop_stop.clicked.connect(self.stop_teleop)
        self.start_enable.clicked.connect(self.start_control_enable_publishing)
        self.stop_enable.clicked.connect(self.stop_control_enable_publishing)

    def connect_publishing_button(self, button, direction, speed_getter):
        button.pressed.connect(
            lambda: self.update_manual_velocity(direction, speed_getter())
        )
        button.released.connect(self.reset_manual_velocity)

    def get_linear_spinbox(self):
        return self.linear_speed_spinbox.value()

    def get_angular_spinbox(self):
        return self.angular_speed_spinbox.value()

    def start_teleop(self):
        cmd = ["roslaunch", "auv_teleop", "start_teleop.launch"]
        if self.xbox_check.isChecked():
            cmd.append("controller:=xbox")
        print(f"Executing: {' '.join(cmd)}")
        self.launch_process = subprocess.Popen(cmd)

    def stop_teleop(self):
        if self.launch_process is not None:
            print("Terminating teleop process...")
            self.launch_process.terminate()
            try:
                self.launch_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("Process did not terminate, killing it...")
                self.launch_process.kill()
            self.launch_process = None
        else:
            print("No teleop process to stop.")

    def publish_velocity_loop(self):
        rate = rospy.Rate(20)
        while self.publishing and not rospy.is_shutdown():
            self.cmd_vel_pub.publish(self.current_twist)
            rate.sleep()

    def update_manual_velocity(self, direction, speed):
        self.publishing = True
        self.current_twist = Twist()

        direction_mapping = {
            "forward": lambda: setattr(self.current_twist.linear, "x", speed),
            "backward": lambda: setattr(self.current_twist.linear, "x", -speed),
            "left": lambda: setattr(self.current_twist.linear, "y", speed),
            "right": lambda: setattr(self.current_twist.linear, "y", -speed),
            "pos_z": lambda: setattr(self.current_twist.linear, "z", speed),
            "neg_z": lambda: setattr(self.current_twist.linear, "z", -speed),
            "pos_yaw": lambda: setattr(self.current_twist.angular, "z", speed),
            "neg_yaw": lambda: setattr(self.current_twist.angular, "z", -speed),
        }

        action = direction_mapping.get(direction)
        if action:
            action()
        else:
            print(f"Unknown direction: {direction}")

        if not hasattr(self, "publish_thread") or not self.publish_thread.is_alive():
            self.publish_thread = threading.Thread(target=self.publish_velocity_loop)
            self.publish_thread.start()

    def reset_manual_velocity(self):
        self.publishing = False
        self.current_twist = Twist()
        self.cmd_vel_pub.publish(self.current_twist)

    def publish_control_enable_loop(self):
        rate = rospy.Rate(20)
        while self.enable_publishing and not rospy.is_shutdown():
            self.enable_pub.publish(True)
            rate.sleep()

    def start_control_enable_publishing(self):
        if not self.enable_publishing:
            self.enable_publishing = True
            self.enable_thread = threading.Thread(
                target=self.publish_control_enable_loop
            )
            self.enable_thread.start()

    def stop_control_enable_publishing(self):
        self.enable_publishing = False
        if self.enable_thread and self.enable_thread.is_alive():
            self.enable_thread.join()
        self.enable_pub.publish(False)

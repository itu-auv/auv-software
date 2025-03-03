#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
    QSizePolicy,
)
import subprocess


class VehicleControlTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QGridLayout()

        # Teleop Section
        teleop_group = QGroupBox("Teleoperation")
        teleop_layout = QHBoxLayout()
        self.teleop_start = QPushButton("Start Teleop")
        self.xbox_check = QCheckBox("Xbox")
        self.teleop_stop = QPushButton("Stop Teleop")
        teleop_layout.addWidget(self.teleop_start)
        teleop_layout.addWidget(self.xbox_check)
        teleop_layout.addWidget(self.teleop_stop)
        teleop_group.setLayout(teleop_layout)

        # Manual Control
        control_group = QGroupBox("Manual Control")
        control_layout = QGridLayout()

        # Button sizes
        button_size = 60

        # Movement Buttons
        self.forward_btn = QPushButton("^")
        self.forward_btn.setFixedSize(button_size, button_size)
        self.forward_btn.pressed.connect(lambda: self.send_cmd_vel("forward", 0.5))
        self.forward_btn.released.connect(lambda: self.send_cmd_vel("forward", 0.0))

        self.left_btn = QPushButton("<")
        self.left_btn.setFixedSize(button_size, button_size)
        self.left_btn.pressed.connect(lambda: self.send_cmd_vel("left", 0.5))
        self.left_btn.released.connect(lambda: self.send_cmd_vel("left", 0.0))

        self.right_btn = QPushButton(">")
        self.right_btn.setFixedSize(button_size, button_size)
        self.right_btn.pressed.connect(lambda: self.send_cmd_vel("right", 0.5))
        self.right_btn.released.connect(lambda: self.send_cmd_vel("right", 0.0))

        self.backward_btn = QPushButton("v")
        self.backward_btn.setFixedSize(button_size, button_size)
        self.backward_btn.pressed.connect(lambda: self.send_cmd_vel("backward", 0.5))
        self.backward_btn.released.connect(lambda: self.send_cmd_vel("backward", 0.0))

        self.up_btn = QPushButton("Up")
        self.up_btn.setFixedSize(button_size, button_size)
        self.up_btn.pressed.connect(lambda: self.send_cmd_vel("pos_z", 0.4))
        self.up_btn.released.connect(lambda: self.send_cmd_vel("pos_z", 0.0))

        self.down_btn = QPushButton("Down")
        self.down_btn.setFixedSize(button_size, button_size)
        self.down_btn.pressed.connect(lambda: self.send_cmd_vel("neg_z", 0.4))
        self.down_btn.released.connect(lambda: self.send_cmd_vel("neg_z", 0.0))

        self.yaw_left_btn = QPushButton("Left")
        self.yaw_left_btn.setFixedSize(button_size, button_size)
        self.yaw_left_btn.pressed.connect(lambda: self.send_cmd_vel("pos_yaw", 0.3))
        self.yaw_left_btn.released.connect(lambda: self.send_cmd_vel("pos_yaw", 0.0))

        self.yaw_right_btn = QPushButton("Right")
        self.yaw_right_btn.setFixedSize(button_size, button_size)
        self.yaw_right_btn.pressed.connect(lambda: self.send_cmd_vel("neg_yaw", 0.3))
        self.yaw_right_btn.released.connect(lambda: self.send_cmd_vel("neg_yaw", 0.0))

        # Button positions
        control_layout.addWidget(self.forward_btn, 0, 1)
        control_layout.addWidget(self.left_btn, 1, 0)
        control_layout.addWidget(self.right_btn, 1, 2)
        control_layout.addWidget(self.backward_btn, 2, 1)
        control_layout.addWidget(self.up_btn, 0, 4)
        control_layout.addWidget(self.down_btn, 2, 4)
        control_layout.addWidget(self.yaw_left_btn, 1, 3)
        control_layout.addWidget(self.yaw_right_btn, 1, 5)

        control_group.setLayout(control_layout)

        layout.addWidget(teleop_group, 0, 0, 1, 2)
        layout.addWidget(control_group, 1, 0, 1, 2)
        self.setLayout(layout)

        # Connections
        self.teleop_start.clicked.connect(self.start_teleop)
        self.teleop_stop.clicked.connect(self.stop_teleop)
        # Add movement button connections...

    def start_teleop(self):
        cmd = "roslaunch auv_teleop start_teleop.launch"
        if self.xbox_check.isChecked():
            cmd = "roslaunch auv_teleop start_teleop.launch controller:=xbox"
        print(f"Executing: {cmd}")
        subprocess.Popen(cmd, shell=True)

    def stop_teleop(self):
        subprocess.Popen("rosnode kill /taluy/joy_node", shell=True)
        subprocess.Popen("rosnode kill /taluy/joystick_node", shell=True)

    def send_cmd_vel(self, direction, speed):
        if direction == "forward":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: {speed}
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"'''
        elif direction == "backward":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: {-speed}
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"'''
        elif direction == "left":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: 0.0
  y: {speed}
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"'''
        elif direction == "right":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: 0.0
  y: {-speed}
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"'''
        elif direction == "pos_yaw":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: {speed}"'''
        elif direction == "neg_yaw":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: {-speed}"'''
        elif direction == "pos_z":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: 0.0
  y: 0.0
  z: {speed}
angular:
  x: 0.0
  y: 0.0
  z: 0.0"'''
        elif direction == "neg_z":
            command = f'''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
  x: 0.0
  y: 0.0
  z: {-speed}
angular:
  x: 0.0
  y: 0.0
  z: 0.0"'''

        subprocess.Popen(command, shell=True)

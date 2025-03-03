#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QDoubleSpinBox,
    QHBoxLayout,
    QCheckBox,
    QLabel,
)
from PyQt5.QtCore import Qt
import subprocess


class ServicesTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Depth Control
        depth_group = QGroupBox("Depth Control")
        depth_layout = QHBoxLayout()
        self.depth_spin = QDoubleSpinBox()
        self.depth_spin.setRange(-3.0, 0.0)
        self.depth_spin.setValue(-1.0)
        self.depth_spin.setSingleStep(0.1)
        self.depth_spin.setDecimals(1)
        self.set_depth_btn = QPushButton("Set Depth")
        depth_layout.addWidget(QLabel("Target Depth:"))
        depth_layout.addWidget(self.depth_spin)
        depth_layout.addWidget(self.set_depth_btn)
        depth_group.setLayout(depth_layout)

        # Button sizes
        button_size_one = 160
        button_size_two = 75

        # Services
        service_group = QGroupBox("Services")
        service_layout = QHBoxLayout()
        self.localization_btn = QPushButton("Start Localization")
        self.localization_btn.setFixedSize(button_size_one, button_size_two)
        self.dvl_btn = QPushButton("Enable DVL")
        self.dvl_btn.setFixedSize(button_size_one, button_size_two)
        # Add the localization button and stretch to create space
        service_layout.addWidget(self.localization_btn, 0, Qt.AlignLeft)
        service_layout.addStretch(1)
        # Add the DVL button
        service_layout.addWidget(self.dvl_btn, 0, Qt.AlignLeft)
        service_group.setLayout(service_layout)

        # Missions
        mission_group = QGroupBox("Missions")
        mission_layout = QHBoxLayout()
        self.mission_toggle = QCheckBox("Enable")
        self.drop_ball_btn = QPushButton("Drop Ball")
        self.torpedo1_btn = QPushButton("Fire Torpedo 1")
        self.torpedo2_btn = QPushButton("Fire Torpedo 2")
        mission_layout.addWidget(self.mission_toggle)
        mission_layout.addWidget(self.drop_ball_btn)
        mission_layout.addWidget(self.torpedo1_btn)
        mission_layout.addWidget(self.torpedo2_btn)
        mission_group.setLayout(mission_layout)

        layout.addWidget(depth_group)
        layout.addWidget(service_group)
        layout.addWidget(mission_group)
        self.setLayout(layout)

        # Initial state
        self.toggle_missions(False)

        # Connections
        self.set_depth_btn.clicked.connect(self.set_depth)
        self.mission_toggle.stateChanged.connect(self.toggle_missions)
        self.drop_ball_btn.clicked.connect(self.drop_ball)
        self.localization_btn.clicked.connect(self.start_localization)
        self.dvl_btn.clicked.connect(self.enable_dvl)
        self.torpedo1_btn.clicked.connect(self.launch_torpedo_1)
        self.torpedo2_btn.clicked.connect(self.launch_torpedo_2)

    def set_depth(self):
        depth = self.depth_spin.value()
        command = f'rosservice call /taluy/set_depth "target_depth: {depth}"'
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def toggle_missions(self, state):
        enable = state == Qt.Checked
        self.drop_ball_btn.setEnabled(enable)
        self.torpedo1_btn.setEnabled(enable)
        self.torpedo2_btn.setEnabled(enable)

    def start_localization(self):
        command = 'rosservice call /taluy/auv_localization_node/enable "{}"'
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def enable_dvl(self):
        command = 'rosservice call /taluy/sensors/dvl/enable "data: true"'
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def drop_ball(self):
        command = "rosservice call /taluy/actuators/ball_dropper/drop"
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def launch_torpedo_1(self):
        command = "rosservice call /taluy/actuators/torpedo_1/launch"
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def launch_torpedo_2(self):
        command = "rosservice call /taluy/actuators/torpedo_2/launch"
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

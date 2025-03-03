#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
)
from PyQt5.QtCore import Qt
import subprocess


class SimulationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Detection Section
        detect_group = QGroupBox("Object Detection")
        detect_layout = QHBoxLayout()
        self.detect_start = QPushButton("Start Detection")
        self.detect_stop = QPushButton("Stop Detection")
        detect_layout.addWidget(self.detect_start)
        detect_layout.addWidget(self.detect_stop)
        detect_group.setLayout(detect_layout)

        # SMACH Section
        smach_group = QGroupBox("State Machine")
        smach_layout = QVBoxLayout()

        # Control Row
        control_row = QHBoxLayout()
        self.smach_start = QPushButton("Launch SMACH")
        self.test_check = QCheckBox("Test Mode")
        self.smach_stop = QPushButton("Stop SMACH")
        control_row.addWidget(self.smach_start)
        control_row.addWidget(self.test_check)
        control_row.addWidget(self.smach_stop)

        # State Checkboxes
        state_row = QHBoxLayout()
        self.states = ["init", "gate", "buoy", "torpedo", "bin", "octagon"]
        self.state_checks = {state: QCheckBox(state) for state in self.states}
        for cb in self.state_checks.values():
            state_row.addWidget(cb)

        smach_layout.addLayout(control_row)
        smach_layout.addLayout(state_row)
        smach_group.setLayout(smach_layout)

        layout.addWidget(detect_group)
        layout.addWidget(smach_group)
        self.setLayout(layout)

        # Connections
        self.detect_start.clicked.connect(
            lambda: subprocess.Popen(
                "roslaunch auv_detection tracker.launch", shell=True
            )
        )
        self.detect_stop.clicked.connect(
            lambda: subprocess.Popen("rosnode kill /tracker_node", shell=True)
        )
        self.smach_start.clicked.connect(self.start_smach)
        self.smach_stop.clicked.connect(
            lambda: subprocess.Popen("rosnode kill /main_state_machine", shell=True)
        )
        self.test_check.stateChanged.connect(self.toggle_state_checks)

        # Initial state
        self.toggle_state_checks(Qt.Unchecked)

    def toggle_state_checks(self, state):
        enabled = state == Qt.Checked
        for cb in self.state_checks.values():
            cb.setEnabled(enabled)
            if not enabled:
                cb.setChecked(False)

    def start_smach(self):
        cmd = "roslaunch auv_smach start.launch"
        if self.test_check.isChecked():
            states = ",".join(
                [s for s, cb in self.state_checks.items() if cb.isChecked()]
            )
            cmd += f" test_mode:=true test_states:={states}"
        print(f"Executing: {cmd}")
        subprocess.Popen(cmd, shell=True)
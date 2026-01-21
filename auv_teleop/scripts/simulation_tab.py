#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
    QComboBox,
    QLabel,
)
from PyQt5.QtCore import Qt
import subprocess
import rospy
from std_msgs.msg import Bool


class SimulationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.detect_process = None
        self.smach_process = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        detect_group = QGroupBox("Object Detection")
        detect_layout = QHBoxLayout()
        self.detect_start = QPushButton("Start Detection")
        self.detect_stop = QPushButton("Stop Detection")
        detect_layout.addWidget(self.detect_start)
        detect_layout.addWidget(self.detect_stop)
        detect_group.setLayout(detect_layout)

        smach_group = QGroupBox("State Machine")
        smach_layout = QVBoxLayout()

        # Competition selector row
        competition_row = QHBoxLayout()
        competition_label = QLabel("Competition:")
        self.competition_combo = QComboBox()
        self.competition_combo.addItems(["RoboSub", "TAC"])
        competition_row.addWidget(competition_label)
        competition_row.addWidget(self.competition_combo)
        competition_row.addStretch()

        # TAC task selector (initially hidden)
        self.tac_task_label = QLabel("Task:")
        self.tac_task_combo = QComboBox()
        self.tac_task_combo.addItems(["docking"])
        competition_row.addWidget(self.tac_task_label)
        competition_row.addWidget(self.tac_task_combo)
        self.tac_task_label.hide()
        self.tac_task_combo.hide()

        # Control row
        control_row = QHBoxLayout()
        self.smach_start = QPushButton("Launch SMACH")
        self.test_check = QCheckBox("Test Mode")
        self.smach_stop = QPushButton("Stop SMACH")
        control_row.addWidget(self.smach_start)
        control_row.addWidget(self.test_check)
        control_row.addWidget(self.smach_stop)

        # RoboSub state checkboxes
        self.robosub_state_row = QHBoxLayout()
        self.robosub_states = ["init", "gate", "slalom", "torpedo", "bin", "octagon"]
        self.robosub_state_checks = {
            state: QCheckBox(state) for state in self.robosub_states
        }
        for cb in self.robosub_state_checks.values():
            self.robosub_state_row.addWidget(cb)

        # TAC state checkboxes (initially hidden)
        self.tac_state_row = QHBoxLayout()
        self.tac_states = ["init", "docking"]
        self.tac_state_checks = {state: QCheckBox(state) for state in self.tac_states}
        for cb in self.tac_state_checks.values():
            self.tac_state_row.addWidget(cb)

        # Wrapper widgets for showing/hiding state rows
        self.robosub_state_widget = QWidget()
        self.robosub_state_widget.setLayout(self.robosub_state_row)

        self.tac_state_widget = QWidget()
        self.tac_state_widget.setLayout(self.tac_state_row)
        self.tac_state_widget.hide()

        smach_layout.addLayout(competition_row)
        smach_layout.addLayout(control_row)
        smach_layout.addWidget(self.robosub_state_widget)
        smach_layout.addWidget(self.tac_state_widget)
        smach_group.setLayout(smach_layout)

        layout.addWidget(detect_group)
        layout.addWidget(smach_group)
        self.setLayout(layout)

        # Connect signals
        self.detect_start.clicked.connect(self.start_detection)
        self.detect_stop.clicked.connect(self.stop_detection)
        self.smach_start.clicked.connect(self.start_smach)
        self.smach_stop.clicked.connect(self.stop_smach)
        self.test_check.stateChanged.connect(self.toggle_state_checks)
        self.competition_combo.currentTextChanged.connect(self.on_competition_changed)

        self.toggle_state_checks(Qt.Unchecked)

    def on_competition_changed(self, competition):
        if competition == "RoboSub":
            self.robosub_state_widget.show()
            self.tac_state_widget.hide()
            self.tac_task_label.hide()
            self.tac_task_combo.hide()
        else:  # TAC
            self.robosub_state_widget.hide()
            self.tac_state_widget.show()
            self.tac_task_label.show()
            self.tac_task_combo.show()
        # Re-apply test mode state
        self.toggle_state_checks(
            Qt.Checked if self.test_check.isChecked() else Qt.Unchecked
        )

    def toggle_state_checks(self, state):
        enabled = state == Qt.Checked
        # Toggle RoboSub state checks
        for cb in self.robosub_state_checks.values():
            cb.setEnabled(enabled)
            if not enabled:
                cb.setChecked(False)
        # Toggle TAC state checks
        for cb in self.tac_state_checks.values():
            cb.setEnabled(enabled)
            if not enabled:
                cb.setChecked(False)

    def start_detection(self):
        cmd = ["roslaunch", "auv_detection", "tracker.launch", "device:=cpu"]
        print(f"Executing: {' '.join(cmd)}")
        self.detect_process = subprocess.Popen(cmd)

    def stop_detection(self):
        if self.detect_process is not None:
            print("Terminating detection process...")
            self.detect_process.terminate()
            try:
                self.detect_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("Detection process did not terminate, killing it...")
                self.detect_process.kill()
            self.detect_process = None
        else:
            print("No detection process to stop.")

    def start_smach(self):
        competition = self.competition_combo.currentText().lower()
        cmd = [
            "roslaunch",
            "auv_smach",
            "start.launch",
            f"competition:={competition}",
            "sim:=true",
            "device:=cpu",
        ]

        if competition == "robosub":
            if self.test_check.isChecked():
                states = ",".join(
                    [
                        s
                        for s, cb in self.robosub_state_checks.items()
                        if cb.isChecked()
                    ]
                )
                cmd.append("test_mode:=true")
                cmd.append(f"test_states:={states}")
                cmd.append("roll:=false")
        else:  # TAC
            tac_task = self.tac_task_combo.currentText()
            cmd.append(f"tac_task:={tac_task}")
            if self.test_check.isChecked():
                states = ",".join(
                    [s for s, cb in self.tac_state_checks.items() if cb.isChecked()]
                )
                cmd.append("test_mode:=true")
                cmd.append(f"test_states:={states}")
            else:
                # TAC runs in test mode by default (init + selected task)
                cmd.append("test_mode:=true")
                cmd.append(f"test_states:=init,{tac_task}")

        print(f"Executing: {' '.join(cmd)}")
        self.smach_process = subprocess.Popen(cmd)

    def stop_smach(self):
        if self.smach_process is not None:
            print("Terminating SMACH process...")
            self.smach_process.terminate()
            try:
                self.smach_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                print("SMACH process did not terminate, killing it...")
                self.smach_process.kill()
            self.smach_process = None
        else:
            print("No SMACH process to stop.")

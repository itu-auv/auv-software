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

        rqt_layout = QHBoxLayout()
        self.rqt_btn = QPushButton("Open rqt_image_view")
        self.rqt_btn.setStyleSheet("background-color: lightblue; color: black;")
        rqt_layout.addWidget(self.rqt_btn)
        layout.addLayout(rqt_layout)

        detect_group = QGroupBox("Object Detection")
        detect_layout = QHBoxLayout()
        self.detect_start = QPushButton("Start Detection")
        self.detect_start.setStyleSheet("background-color: lightgreen; color: black;")
        self.detect_stop = QPushButton("Stop Detection")
        self.detect_stop.setStyleSheet("background-color: lightsalmon; color: black;")
        detect_layout.addWidget(self.detect_start)
        detect_layout.addWidget(self.detect_stop)
        detect_group.setLayout(detect_layout)

        smach_group = QGroupBox("State Machine")
        smach_layout = QVBoxLayout()

        control_row = QHBoxLayout()
        self.smach_start = QPushButton("Launch SMACH")
        self.smach_start.setStyleSheet("background-color: lightgreen; color: black;")
        self.test_check = QCheckBox("Test Mode")
        self.smach_stop = QPushButton("Stop SMACH")
        self.smach_stop.setStyleSheet("background-color: lightsalmon; color: black;")
        control_row.addWidget(self.smach_start)
        control_row.addWidget(self.test_check)
        control_row.addWidget(self.smach_stop)

        state_row = QHBoxLayout()
        self.states = ["init", "gate", "buoy", "torpedo", "bin", "octagon"]
        self.state_checks = {state: QCheckBox(state) for state in self.states}
        for cb in self.state_checks.values():
            state_row.addWidget(cb)

        smach_layout.addLayout(control_row)
        smach_layout.addLayout(state_row)
        smach_group.setLayout(smach_layout)

        propulsion_group = QGroupBox("Propulsion Board")
        propulsion_layout = QHBoxLayout()
        self.propulsion_btn = QPushButton("Publish propulsion_board")
        self.propulsion_btn.setStyleSheet("background-color: lightgreen; color: black;")
        propulsion_layout.addWidget(self.propulsion_btn)
        propulsion_group.setLayout(propulsion_layout)

        layout.addWidget(detect_group)
        layout.addWidget(smach_group)
        layout.addWidget(propulsion_group)
        self.setLayout(layout)

        self.propulsion_pub = rospy.Publisher(
            "propulsion_board/status", Bool, queue_size=10
        )

        self.rqt_btn.clicked.connect(self.open_rqt)
        self.detect_start.clicked.connect(self.start_detection)
        self.detect_stop.clicked.connect(self.stop_detection)
        self.smach_start.clicked.connect(self.start_smach)
        self.smach_stop.clicked.connect(self.stop_smach)
        self.test_check.stateChanged.connect(self.toggle_state_checks)
        self.propulsion_btn.clicked.connect(self.publish_propulsion_board)

        self.toggle_state_checks(Qt.Unchecked)

    def toggle_state_checks(self, state):
        enabled = state == Qt.Checked
        for cb in self.state_checks.values():
            cb.setEnabled(enabled)
            if not enabled:
                cb.setChecked(False)

    def start_detection(self):
        cmd = ["roslaunch", "auv_detection", "tracker.launch"]
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
        cmd = ["roslaunch", "auv_smach", "start.launch"]
        if self.test_check.isChecked():
            states = ",".join(
                [s for s, cb in self.state_checks.items() if cb.isChecked()]
            )
            cmd.append(f"test_mode:=true")
            cmd.append(f"test_states:={states}")
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

    def open_rqt(self):
        subprocess.Popen(["rqt", "-s", "rqt_image_view"])

    def publish_propulsion_board(self):
        msg = Bool()
        msg.data = True
        self.propulsion_pub.publish(msg)

#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
)
from command_thread import CommandThread
import subprocess


class DryTestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.imu_thread = None
        self.bar30_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Sensor Monitoring
        sensor_group = QGroupBox("Sensor Monitoring")
        sensor_layout = QVBoxLayout()

        # IMU
        imu_layout = QHBoxLayout()
        self.imu_start_btn = QPushButton("Start IMU Echo")
        self.imu_stop_btn = QPushButton("Stop IMU Echo")
        imu_layout.addWidget(self.imu_start_btn)
        imu_layout.addWidget(self.imu_stop_btn)

        # Bar30
        bar30_layout = QHBoxLayout()
        self.bar30_start_btn = QPushButton("Start Bar30 Echo")
        self.bar30_stop_btn = QPushButton("Stop Bar30 Echo")
        bar30_layout.addWidget(self.bar30_start_btn)
        bar30_layout.addWidget(self.bar30_stop_btn)

        sensor_layout.addLayout(imu_layout)
        sensor_layout.addLayout(bar30_layout)
        sensor_group.setLayout(sensor_layout)

        # Output
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        # Image View
        self.rqt_btn = QPushButton("Open rqt_image_view")

        layout.addWidget(sensor_group)
        layout.addWidget(self.rqt_btn)
        layout.addWidget(self.output)
        self.setLayout(layout)

        # Connections
        self.imu_start_btn.clicked.connect(self.start_imu)
        self.imu_stop_btn.clicked.connect(self.stop_imu)
        self.bar30_start_btn.clicked.connect(self.start_bar30)
        self.bar30_stop_btn.clicked.connect(self.stop_bar30)
        self.rqt_btn.clicked.connect(self.open_rqt)

    def start_imu(self):
        self.imu_thread = CommandThread("rostopic echo /taluy/sensors/imu/data")
        self.imu_thread.output_signal.connect(self.output.append)
        self.imu_thread.start()

    def stop_imu(self):
        if self.imu_thread:
            self.imu_thread.stop()

    def start_bar30(self):
        self.bar30_thread = CommandThread(
            "rostopic echo /taluy/sensors/external_pressure_sensor/depth"
        )
        self.bar30_thread.output_signal.connect(self.output.append)
        self.bar30_thread.start()

    def stop_bar30(self):
        if self.bar30_thread:
            self.bar30_thread.stop()

    def open_rqt(self):
        subprocess.Popen("rqt_image_view", shell=True)

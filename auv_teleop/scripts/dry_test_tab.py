#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QTextEdit,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QCheckBox,
)
from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class CommandThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._is_running = True

    def run(self):
        process = subprocess.Popen(
            self.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while self._is_running:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                self.output_signal.emit(output.strip())
        process.stdout.close()
        process.wait()

    def stop(self):
        self._is_running = False


class DryTestTab(QWidget):
    def __init__(self):
        super().__init__()

        self.topic_imu = rospy.get_param("~topic_imu", "imu/data")
        self.topic_pressure = rospy.get_param("~topic_pressure", "depth")
        self.topic_camera_bottom = rospy.get_param(
            "~topic_camera_bottom", "cam_bottom/image_raw"
        )
        self.topic_camera_front = rospy.get_param(
            "~topic_camera_front", "cam_front/image_raw"
        )

        self.cmd_vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=10)
        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=10)
        self.topic_imu = rospy.get_param("~topic_imu", "imu/data")
        self.topic_pressure = rospy.get_param("~topic_pressure", "depth")
        self.topic_camera_bottom = rospy.get_param(
            "~topic_camera_bottom", "cam_bottom/image_raw"
        )
        self.topic_camera_front = rospy.get_param(
            "~topic_camera_front", "cam_front/image_raw"
        )
        self.init_ui()
        self.imu_thread = None
        self.bar30_thread = None
        self.publishing = False
        self.current_twist = Twist()
        self.enable_thread = None
        self.enable_publishing = False
        self.launch_process = None

    def init_ui(self):
        layout = QGridLayout()

        sensor_group = QGroupBox("Sensor Monitoring")
        sensor_layout = QGridLayout()

        self.imu_start_btn = QPushButton("Echo IMU")
        self.imu_stop_btn = QPushButton("Stop Echo IMU")
        self.bar30_start_btn = QPushButton("Echo Bar30")
        self.bar30_stop_btn = QPushButton("Stop Echo Bar30")

        sensor_layout.addWidget(self.imu_start_btn, 0, 0)
        sensor_layout.addWidget(self.imu_stop_btn, 0, 1)
        sensor_layout.addWidget(self.bar30_start_btn, 1, 0)
        sensor_layout.addWidget(self.bar30_stop_btn, 1, 1)
        sensor_group.setLayout(sensor_layout)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        button_layout = QVBoxLayout()
        self.dry_test_btn = QPushButton("Run Dry Test")
        self.dry_test_btn.setStyleSheet("background-color: green; color: white;")

        self.rqt_btn = QPushButton("Open rqt_image_view")
        self.rqt_btn.setStyleSheet("background-color: lightblue; color: black;")

        button_layout.addWidget(self.dry_test_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.rqt_btn)

        layout.addWidget(sensor_group, 0, 0)
        layout.addLayout(button_layout, 1, 0)
        layout.addWidget(self.output, 2, 0)

        self.clear_btn = QPushButton("Clear")
        layout.addWidget(self.clear_btn, 3, 0)

        # Teleoperation controls
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

        layout.addWidget(teleop_group, 4, 0)

        self.setLayout(layout)

        self.teleop_start.clicked.connect(self.start_teleop)
        self.teleop_stop.clicked.connect(self.stop_teleop)
        self.start_enable.clicked.connect(self.start_control_enable_publishing)
        self.stop_enable.clicked.connect(self.stop_control_enable_publishing)

        self.imu_start_btn.clicked.connect(self.start_imu)
        self.imu_stop_btn.clicked.connect(self.stop_imu)
        self.bar30_start_btn.clicked.connect(self.start_bar30)
        self.bar30_stop_btn.clicked.connect(self.stop_bar30)
        self.rqt_btn.clicked.connect(self.open_rqt)
        self.dry_test_btn.clicked.connect(self.run_dry_test)
        self.clear_btn.clicked.connect(self.clear_output)

    def start_imu(self):
        cmd = f"rostopic echo {self.topic_imu}"
        self.output.append(f"Running: {cmd}")
        self.imu_thread = CommandThread(cmd)
        self.imu_thread.output_signal.connect(self.output.append)
        self.imu_thread.start()

    def stop_imu(self):
        if self.imu_thread:
            self.imu_thread.stop()

    def start_bar30(self):
        cmd = f"rostopic echo {self.topic_pressure}"
        self.output.append(f"Running: {cmd}")
        self.bar30_thread = CommandThread(cmd)
        self.bar30_thread.output_signal.connect(self.output.append)
        self.bar30_thread.start()

    def stop_bar30(self):
        if self.bar30_thread:
            self.bar30_thread.stop()

    def open_rqt(self):
        subprocess.Popen("rqt -s rqt_image_view", shell=True)

    def run_dry_test(self):
        topics = {
            self.topic_camera_bottom: "Bottom Camera",
            self.topic_camera_front: "Front Camera",
            self.topic_imu: "IMU",
            self.topic_pressure: "Bar30",
        }

        active_topics = rospy.get_published_topics()
        active_topics = [t[0] for t in active_topics]

        failed_topics = []
        for topic, name in topics.items():
            if not topic.startswith("/"):
                full_topic = "/" + topic
            else:
                full_topic = topic

            if full_topic not in active_topics:
                failed_topics.append(f"{name}")

        if not failed_topics:
            self.output.append("Dry test is successful!")
        else:
            self.output.append(
                "Dry test failed! Issues detected in:\n" + "\n".join(failed_topics)
            )

    def clear_output(self):
        self.output.clear()

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

    def start_control_enable_publishing(self):
        self.enable_publishing = True
        self.enable_thread = threading.Thread(target=self.publish_enable)
        self.enable_thread.start()

    def stop_control_enable_publishing(self):
        self.enable_publishing = False
        if self.enable_thread is not None:
            self.enable_thread.join()
            self.enable_thread = None

    def publish_enable(self):
        rate = rospy.Rate(20)
        while self.enable_publishing and not rospy.is_shutdown():
            self.enable_pub.publish(True)
            rate.sleep()

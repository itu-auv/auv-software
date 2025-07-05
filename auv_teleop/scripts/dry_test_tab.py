#!/usr/bin/env python3

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QTextEdit,
    QGridLayout,
    QCheckBox,
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import subprocess
import rospy
from std_msgs.msg import Bool
import auv_msgs.msg
import threading


class CommandThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._is_running = True
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            while self._is_running:
                output = self.process.stdout.readline()
                if output == "" and self.process.poll() is not None:
                    break
                if output:
                    self.output_signal.emit(output.strip())
        finally:
            self.cleanup()

    def cleanup(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            finally:
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()
                self.process = None

    def stop(self):
        self._is_running = False
        self.cleanup()


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

        self.enable_pub = rospy.Publisher("enable", Bool, queue_size=10)
        self.thruster_pub = rospy.Publisher(
            "drive_pulse", auv_msgs.msg.MotorCommand, queue_size=10
        )

        self.init_ui()
        self.imu_thread = None
        self.bar30_thread = None
        self.enable_thread = None
        self.enable_publishing = False
        self.launch_process = None
        self.thruster_test_timer = QTimer()
        self.thruster_test_timer.timeout.connect(self.stop_thruster_test)

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

        self.thruster_test_btn = QPushButton("Test Thrusters")
        self.thruster_test_btn.setStyleSheet("background-color: orange; color: black;")

        button_layout.addWidget(self.dry_test_btn)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.rqt_btn)
        button_layout.addWidget(self.thruster_test_btn)

        layout.addWidget(sensor_group, 0, 0)
        layout.addLayout(button_layout, 1, 0)
        layout.addWidget(self.output, 2, 0)

        self.clear_btn = QPushButton("Clear")
        layout.addWidget(self.clear_btn, 3, 0)

        # Teleoperation controls
        teleop_group = QGroupBox("Teleoperation")
        teleop_layout = QGridLayout()
        self.teleop_start = QPushButton("Start Teleop")
        self.enable_control = QPushButton("Enable Control")
        self.disable_control = QPushButton("Disable Control")
        self.xbox_check = QCheckBox("Xbox")
        self.teleop_stop = QPushButton("Stop Teleop")
        teleop_layout.addWidget(self.teleop_start, 0, 0)
        teleop_layout.addWidget(self.enable_control, 1, 0)
        teleop_layout.addWidget(self.disable_control, 1, 2)
        teleop_layout.addWidget(self.xbox_check, 0, 1)
        teleop_layout.addWidget(self.teleop_stop, 0, 2)
        teleop_group.setLayout(teleop_layout)

        layout.addWidget(teleop_group, 4, 0)

        self.setLayout(layout)

        self.teleop_start.clicked.connect(self.start_teleop)
        self.teleop_stop.clicked.connect(self.stop_teleop)
        self.enable_control.clicked.connect(self.start_control_enable_publishing)
        self.disable_control.clicked.connect(self.stop_control_enable_publishing)
        self.thruster_test_btn.clicked.connect(self.test_thrusters)

        self.imu_start_btn.clicked.connect(self.start_imu)
        self.imu_stop_btn.clicked.connect(self.stop_imu)
        self.bar30_start_btn.clicked.connect(self.start_bar30)
        self.bar30_stop_btn.clicked.connect(self.stop_bar30)
        self.rqt_btn.clicked.connect(self.open_rqt)
        self.dry_test_btn.clicked.connect(self.run_dry_test)
        self.clear_btn.clicked.connect(self.clear_output)

    def start_imu(self):
        if self.imu_thread and self.imu_thread.isRunning():
            self.output.append("IMU echo is already running")
            return

        cmd = f"rostopic echo {self.topic_imu}"
        self.output.append(f"Running: {cmd}")
        self.imu_thread = CommandThread(cmd)
        self.imu_thread.output_signal.connect(self.output.append)
        self.imu_thread.start()

    def stop_imu(self):
        if self.imu_thread and self.imu_thread.isRunning():
            self.imu_thread.stop()
            self.imu_thread.wait()
            self.imu_thread = None
            self.output.append("IMU echo stopped")

    def start_bar30(self):
        if self.bar30_thread and self.bar30_thread.isRunning():
            self.output.append("Bar30 echo is already running")
            return

        cmd = f"rostopic echo {self.topic_pressure}"
        self.output.append(f"Running: {cmd}")
        self.bar30_thread = CommandThread(cmd)
        self.bar30_thread.output_signal.connect(self.output.append)
        self.bar30_thread.start()

    def stop_bar30(self):
        if self.bar30_thread and self.bar30_thread.isRunning():
            self.bar30_thread.stop()
            self.bar30_thread.wait()
            self.bar30_thread = None
            self.output.append("Bar30 echo stopped")

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

    def test_thrusters(self):
        """Publish motor commands to test thrusters for 1 second"""
        try:
            motor_cmd = auv_msgs.msg.MotorCommand()
            motor_cmd.channels = [1600] * 8

            self.thruster_test_timer.start(1000)
            self.output.append("Thruster test started! Running for 1 second...")

            # Start continuous publishing
            self.thruster_publishing = True
            self.thruster_thread = threading.Thread(
                target=self.publish_thrusters, args=(motor_cmd,)
            )
            self.thruster_thread.start()

        except Exception as e:
            self.output.append(f"Error testing thrusters: {str(e)}")

    def stop_thruster_test(self):
        try:
            self.thruster_test_timer.stop()
            self.thruster_publishing = False

            if self.thruster_thread is not None:
                self.thruster_thread.join()
                self.thruster_thread = None

            stop_cmd = auv_msgs.msg.MotorCommand()
            stop_cmd.channels = [0] * 8
            self.thruster_pub.publish(stop_cmd)

            self.output.append("Thruster test completed.")

        except Exception as e:
            self.output.append(f"Error stopping thrusters: {str(e)}")

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

    def publish_thrusters(self, motor_cmd):
        rate = rospy.Rate(20)  # 20 Hz
        while self.thruster_publishing and not rospy.is_shutdown():
            self.thruster_pub.publish(motor_cmd)
            rate.sleep()

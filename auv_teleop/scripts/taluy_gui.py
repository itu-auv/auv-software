#!/usr/bin/env python3

import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QFormLayout,
    QTabWidget,
    QTextEdit,
    QDoubleSpinBox,
    QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer


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


class AUVControlGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Taluy AUV Control Panel")
        self.setGeometry(100, 100, 600, 800)

        layout = QVBoxLayout()

        # Tab Widget to organize the interface into different sections
        self.tab_widget = QTabWidget()
        self.tab_services = QWidget()
        self.tab_launch = QWidget()
        self.tab_dry_test = QWidget()
        self.tab_vehicle_control = QWidget()

        self.tab_widget.addTab(self.tab_services, "Services")
        self.tab_widget.addTab(self.tab_launch, "Launch")
        self.tab_widget.addTab(self.tab_dry_test, "Dry Test")
        self.tab_widget.addTab(self.tab_vehicle_control, "Vehicle Control")

        layout.addWidget(self.tab_widget)

        # Services Group
        self.services_group = QGroupBox("Services")
        services_layout = (
            QVBoxLayout()
        )  # QVBoxLayout kullanarak dikey boşluk ekleyeceğiz

        # Depth DoubleSpinBox
        self.depth_spinbox = QDoubleSpinBox()
        self.depth_spinbox.setRange(
            0.0, 3.0
        )  # Set the range of the spinbox (0.0 to 3.0)
        self.depth_spinbox.setSingleStep(0.1)  # Set the step size to 0.1
        self.depth_spinbox.setValue(1.0)  # Set the default value to 1.0
        self.depth_spinbox.setDecimals(1)  # Show only one decimal place

        self.depth_button = QPushButton("Set Depth")
        self.depth_button.clicked.connect(self.set_depth)

        # Depth row
        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Target Depth:"))
        depth_row.addWidget(self.depth_spinbox)
        depth_row.addWidget(self.depth_button)

        services_layout.addLayout(depth_row)
        services_layout.addSpacing(20)  # Boşluk ekle

        # Localization Button
        self.localization_button = QPushButton("Start Localization")
        self.localization_button.clicked.connect(self.start_localization)
        services_layout.addWidget(self.localization_button)
        services_layout.addSpacing(20)  # Boşluk ekle

        # DVL Button
        self.dvl_button = QPushButton("Enable DVL")
        self.dvl_button.clicked.connect(self.enable_dvl)
        services_layout.addWidget(self.dvl_button)

        self.services_group.setLayout(services_layout)
        self.tab_services.setLayout(services_layout)

        # Launch Group
        self.launch_group = QGroupBox("Launch")
        launch_layout = QVBoxLayout()  # QVBoxLayout kullanarak dikey boşluk ekleyeceğiz

        # Teleop Button
        self.teleop_button = QPushButton("Start Teleop")
        self.teleop_button.clicked.connect(self.start_teleop)
        launch_layout.addWidget(self.teleop_button)
        launch_layout.addSpacing(20)  # Boşluk ekle

        # Detection Button
        self.detection_button = QPushButton("Start Detection")
        self.detection_button.clicked.connect(self.start_detection)
        launch_layout.addWidget(self.detection_button)

        self.launch_group.setLayout(launch_layout)
        self.tab_launch.setLayout(launch_layout)

        # Dry Test Group
        self.dry_test_group = QGroupBox("Dry Test")
        dry_test_layout = QVBoxLayout()

        # Echo Buttons with Stop
        echo_buttons_layout = QHBoxLayout()
        self.imu_button = QPushButton("Echo IMU")
        self.imu_button.clicked.connect(self.echo_imu)
        self.stop_imu_button = QPushButton("Stop IMU")
        self.stop_imu_button.clicked.connect(self.stop_imu)
        echo_buttons_layout.addWidget(self.imu_button)
        echo_buttons_layout.addWidget(self.stop_imu_button)

        self.bar30_button = QPushButton("Echo Bar30")
        self.bar30_button.clicked.connect(self.echo_bar30)
        self.stop_bar30_button = QPushButton("Stop Bar30")
        self.stop_bar30_button.clicked.connect(self.stop_bar30)
        echo_buttons_layout.addWidget(self.bar30_button)
        echo_buttons_layout.addWidget(self.stop_bar30_button)

        dry_test_layout.addLayout(echo_buttons_layout)
        dry_test_layout.addSpacing(20)  # Boşluk ekle

        self.rqt_image_view_button = QPushButton("Open rqt_image_view")
        self.rqt_image_view_button.clicked.connect(self.start_rqt_image_view)
        dry_test_layout.addWidget(self.rqt_image_view_button)

        # Output log display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMinimumHeight(300)  # Dikeyde büyütme
        dry_test_layout.addWidget(self.output_display)

        self.dry_test_group.setLayout(dry_test_layout)
        self.tab_dry_test.setLayout(dry_test_layout)

        # Vehicle Control (cmd_vel) Layout
        self.vehicle_control_group = QGroupBox("Vehicle Control")
        vehicle_control_layout = QGridLayout()

        # Buton boyutlarını kare yapma
        button_size = 70

        # Forward Button
        self.forward_button = QPushButton("FORWARD")
        self.forward_button.setFixedSize(button_size, button_size)
        self.forward_button.pressed.connect(lambda: self.send_cmd_vel("forward", 0.5))
        self.forward_button.released.connect(lambda: self.send_cmd_vel("forward", 0.0))
        vehicle_control_layout.addWidget(self.forward_button, 0, 1)

        # Left Button
        self.left_button = QPushButton("LEFT")
        self.left_button.setFixedSize(button_size, button_size)
        self.left_button.pressed.connect(lambda: self.send_cmd_vel("left", 0.5))
        self.left_button.released.connect(lambda: self.send_cmd_vel("left", 0.0))
        vehicle_control_layout.addWidget(self.left_button, 1, 0)

        # Right Button
        self.right_button = QPushButton("RIGHT")
        self.right_button.setFixedSize(button_size, button_size)
        self.right_button.pressed.connect(lambda: self.send_cmd_vel("right", 0.5))
        self.right_button.released.connect(lambda: self.send_cmd_vel("right", 0.0))
        vehicle_control_layout.addWidget(self.right_button, 1, 2)

        # Backward Button
        self.backward_button = QPushButton("BACKWARD")
        self.backward_button.setFixedSize(button_size, button_size)
        self.backward_button.pressed.connect(lambda: self.send_cmd_vel("backward", 0.5))
        self.backward_button.released.connect(
            lambda: self.send_cmd_vel("backward", 0.0)
        )
        vehicle_control_layout.addWidget(self.backward_button, 2, 1)

        # Up Button
        self.up_button = QPushButton("UP")
        self.up_button.setFixedSize(button_size, button_size)
        self.up_button.pressed.connect(lambda: self.send_cmd_vel("pos_z", 0.4))
        self.up_button.released.connect(lambda: self.send_cmd_vel("pos_z", 0.0))
        vehicle_control_layout.addWidget(self.up_button, 0, 5)

        # Down Button
        self.down_button = QPushButton("DOWN")
        self.down_button.setFixedSize(button_size, button_size)
        self.down_button.pressed.connect(lambda: self.send_cmd_vel("neg_z", 0.4))
        self.down_button.released.connect(lambda: self.send_cmd_vel("neg_z", 0.0))
        vehicle_control_layout.addWidget(self.down_button, 2, 5)

        # Pos Yaw Button
        self.pos_yaw_button = QPushButton("YAW+")
        self.pos_yaw_button.setFixedSize(button_size, button_size)
        self.pos_yaw_button.pressed.connect(lambda: self.send_cmd_vel("pos_yaw", 0.3))
        self.pos_yaw_button.released.connect(lambda: self.send_cmd_vel("pos_yaw", 0.0))
        vehicle_control_layout.addWidget(self.pos_yaw_button, 1, 4)

        # Neg Yaw Button
        self.neg_yaw_button = QPushButton("YAW-")
        self.neg_yaw_button.setFixedSize(button_size, button_size)
        self.neg_yaw_button.pressed.connect(lambda: self.send_cmd_vel("neg_yaw", 0.3))
        self.neg_yaw_button.released.connect(lambda: self.send_cmd_vel("neg_yaw", 0.0))
        vehicle_control_layout.addWidget(self.neg_yaw_button, 1, 6)

        self.vehicle_control_group.setLayout(vehicle_control_layout)
        self.tab_vehicle_control.setLayout(vehicle_control_layout)

        self.setLayout(layout)

    def set_depth(self):
        depth = self.depth_spinbox.value()  # Get the value directly as a float
        command = f'rosservice call /taluy/set_depth "target_depth: {depth}"'
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def start_localization(self):
        command = 'rosservice call /taluy/auv_localization_node/enable "{}"'
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def enable_dvl(self):
        command = 'rosservice call /taluy/sensors/dvl/enable "data: true"'
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def start_teleop(self):
        command = "roslaunch auv_teleop start_teleop.launch"
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def start_detection(self):
        command = "roslaunch auv_detection tracker.launch"
        print(f"Executing: {command}")
        subprocess.Popen(command, shell=True)

    def echo_imu(self):
        if hasattr(self, "imu_thread"):
            self.imu_thread.stop()
        command = "rostopic echo /taluy/sensors/imu/data"
        self.display_output(command)
        self.imu_thread = CommandThread(command)
        self.imu_thread.output_signal.connect(self.output_display.append)
        self.imu_thread.start()

    def echo_bar30(self):
        if hasattr(self, "bar30_thread"):
            self.bar30_thread.stop()
        command = "rostopic echo taluy/sensors/external_pressure_sensor/depth"
        self.display_output(command)
        self.bar30_thread = CommandThread(command)
        self.bar30_thread.output_signal.connect(self.output_display.append)
        self.bar30_thread.start()

    def stop_imu(self):
        if hasattr(self, "imu_thread"):
            self.imu_thread.stop()
            self.output_display.append("IMU Echo Stopped")

    def stop_bar30(self):
        if hasattr(self, "bar30_thread"):
            self.bar30_thread.stop()
            self.output_display.append("Bar30 Echo Stopped")

    def start_rqt_image_view(self):
        command = "rqt_image_view"
        self.display_output(command)
        subprocess.Popen(command, shell=True)

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

        self.display_output(command)
        subprocess.Popen(command, shell=True)

    def display_output(self, command):
        self.output_display.append(f"Executing: {command}")
        self.output_display.repaint()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AUVControlGUI()
    window.show()
    sys.exit(app.exec_())

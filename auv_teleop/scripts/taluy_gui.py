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
    QTabWidget,
    QTextEdit,
    QDoubleSpinBox,
    QGridLayout,
    QCheckBox,
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
        self.setGeometry(100, 100, 100, 100)

        layout = QVBoxLayout()

        # Tab Widget to organize the interface into different sections
        self.tab_widget = QTabWidget()
        self.tab_services = QWidget()
        self.tab_launch = QWidget()
        self.tab_dry_test = QWidget()
        self.tab_vehicle_control = QWidget()
        self.tab_simulation = QWidget()

        self.tab_widget.addTab(self.tab_services, "Services")
        self.tab_widget.addTab(self.tab_launch, "Launch")
        self.tab_widget.addTab(self.tab_dry_test, "Dry Test")
        self.tab_widget.addTab(self.tab_vehicle_control, "Vehicle Control")
        self.tab_widget.addTab(self.tab_simulation, "Simulation")

        layout.addWidget(self.tab_widget)

        # Services Group
        self.services_group = QGroupBox("Services")
        services_layout = QVBoxLayout()

        # Depth DoubleSpinBox
        self.depth_spinbox = QDoubleSpinBox()
        self.depth_spinbox.setRange(-3.0, 0.0)
        self.depth_spinbox.setSingleStep(0.1)
        self.depth_spinbox.setValue(-1.0)
        self.depth_spinbox.setDecimals(1)

        self.depth_button = QPushButton("Set Depth")
        self.depth_button.clicked.connect(self.set_depth)

        # Depth row
        depth_row = QHBoxLayout()
        depth_row.addWidget(QLabel("Target Depth:"))
        depth_row.addWidget(self.depth_spinbox)
        depth_row.addWidget(self.depth_button)

        services_layout.addLayout(depth_row)
        services_layout.addSpacing(20)

        # Localization Button
        self.localization_button = QPushButton("Start Localization")
        self.localization_button.clicked.connect(self.start_localization)
        services_layout.addWidget(self.localization_button)
        services_layout.addSpacing(20)

        # DVL Button
        self.dvl_button = QPushButton("Enable DVL")
        self.dvl_button.clicked.connect(self.enable_dvl)
        services_layout.addWidget(self.dvl_button)
        services_layout.addSpacing(5)

        # Missions Toggle Button
        self.missions_button = QCheckBox("Missions")
        self.missions_button.setCheckable(True)
        self.missions_button.toggled.connect(self.toggle_missions_buttons)
        services_layout.addWidget(self.missions_button)
        services_layout.addSpacing(5)

        # Drop Ball Button
        self.drop_ball_button = QPushButton("Drop Ball")
        self.drop_ball_button.clicked.connect(self.drop_ball)
        self.drop_ball_button.setEnabled(False)  # Initially disabled
        services_layout.addWidget(self.drop_ball_button)
        services_layout.addSpacing(20)

        # Torpedo 1 Button
        self.torpedo_1_button = QPushButton("Launch Torpedo 1")
        self.torpedo_1_button.clicked.connect(self.launch_torpedo_1)
        self.torpedo_1_button.setEnabled(False)  # Initially disabled
        services_layout.addWidget(self.torpedo_1_button)
        services_layout.addSpacing(20)

        # Torpedo 2 Button
        self.torpedo_2_button = QPushButton("Launch Torpedo 2")
        self.torpedo_2_button.clicked.connect(self.launch_torpedo_2)
        self.torpedo_2_button.setEnabled(False)  # Initially disabled
        services_layout.addWidget(self.torpedo_2_button)

        self.services_group.setLayout(services_layout)
        self.tab_services.setLayout(services_layout)

        # Launch Group
        self.launch_group = QGroupBox("Launch")
        launch_layout = QVBoxLayout()

        # Teleop Button with Checkbox and Stop Button
        teleop_layout = QHBoxLayout()
        self.teleop_button = QPushButton("Start Teleop")
        self.teleop_button.clicked.connect(self.start_teleop)
        teleop_layout.addWidget(self.teleop_button)

        self.xbox_checkbox = QCheckBox("Xbox")
        teleop_layout.addWidget(self.xbox_checkbox)

        self.stop_teleop_button = QPushButton("Stop Teleop")
        self.stop_teleop_button.clicked.connect(self.stop_teleop)
        teleop_layout.addWidget(self.stop_teleop_button)

        launch_layout.addLayout(teleop_layout)
        launch_layout.addSpacing(20)

        self.launch_group.setLayout(launch_layout)
        self.tab_launch.setLayout(launch_layout)

        # Simulation Tab
        simulation_layout = QVBoxLayout()

        # Detection Button with Stop Button
        detection_layout = QHBoxLayout()
        self.detection_button = QPushButton("Start Detection")
        self.detection_button.clicked.connect(self.start_detection)
        detection_layout.addWidget(self.detection_button)

        self.stop_detection_button = QPushButton("Stop Detection")
        self.stop_detection_button.clicked.connect(self.stop_detection)
        detection_layout.addWidget(self.stop_detection_button)

        simulation_layout.addLayout(detection_layout)
        simulation_layout.addSpacing(20)

        # SMACH State Machine Section
        smach_group = QGroupBox("SMACH State Machine")
        smach_layout = QVBoxLayout()

        # SMACH Control Row
        smach_control_layout = QHBoxLayout()
        self.smach_launch_btn = QPushButton("Launch SMACH")
        self.smach_launch_btn.clicked.connect(self.launch_smach)
        self.test_mode_check = QCheckBox("Test Mode")
        self.smach_stop_btn = QPushButton("Stop SMACH")
        self.smach_stop_btn.clicked.connect(self.stop_smach)

        smach_control_layout.addWidget(self.smach_launch_btn)
        smach_control_layout.addWidget(self.test_mode_check)
        smach_control_layout.addWidget(self.smach_stop_btn)
        smach_layout.addLayout(smach_control_layout)

        # State Checkboxes
        self.state_checks = {}
        state_check_layout = QHBoxLayout()
        for state in ["init", "gate", "torpedo", "bin", "octagon"]:
            cb = QCheckBox(state)
            cb.setEnabled(False)
            cb.stateChanged.connect(lambda val, s=state: self.update_states(val, s))
            self.state_checks[state] = cb
            state_check_layout.addWidget(cb)
        smach_layout.addLayout(state_check_layout)

        smach_group.setLayout(smach_layout)
        simulation_layout.addWidget(smach_group)

        self.tab_simulation.setLayout(simulation_layout)

        # Initialize state tracking
        self.test_states_order = []
        self.test_mode_check.stateChanged.connect(self.toggle_test_mode)

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
        dry_test_layout.addSpacing(20)

        self.rqt_image_view_button = QPushButton("Open rqt_image_view")
        self.rqt_image_view_button.clicked.connect(self.start_rqt_image_view)
        dry_test_layout.addWidget(self.rqt_image_view_button)

        # Output log display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMinimumHeight(300)
        dry_test_layout.addWidget(self.output_display)

        self.dry_test_group.setLayout(dry_test_layout)
        self.tab_dry_test.setLayout(dry_test_layout)

        # Vehicle Control (cmd_vel) Layout
        self.vehicle_control_group = QGroupBox("Vehicle Control")
        vehicle_control_layout = QGridLayout()

        # Button sizes
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

        # Dictionary to store subprocesses
        self.processes = {}

    def toggle_missions_buttons(self, checked):
        self.drop_ball_button.setEnabled(checked)
        self.torpedo_1_button.setEnabled(checked)
        self.torpedo_2_button.setEnabled(checked)
        status = "enabled" if checked else "disabled"
        self.output_display.append(f"Missions buttons {status}")

    def toggle_test_mode(self, state):
        enable = state == Qt.Checked
        for cb in self.state_checks.values():
            cb.setEnabled(enable)
        if not enable:
            self.test_states_order.clear()
            for cb in self.state_checks.values():
                cb.setChecked(False)

    def update_states(self, checked, state):
        if checked:
            if state not in self.test_states_order:
                self.test_states_order.append(state)
        else:
            if state in self.test_states_order:
                self.test_states_order.remove(state)

    def launch_smach(self):
        cmd = "roslaunch auv_smach start.launch"
        if self.test_mode_check.isChecked():
            states = ",".join(self.test_states_order)
            cmd += f" test_mode:=true test_states:={states}"
        subprocess.Popen(cmd, shell=True)
        self.output_display.append(f"Launched SMACH: {cmd}")

    def stop_smach(self):
        subprocess.Popen("rosnode kill /main_state_machine", shell=True)
        self.output_display.append("Stopped SMACH state machine")

    def set_depth(self):
        depth = self.depth_spinbox.value()
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

    def start_teleop(self):
        if self.xbox_checkbox.isChecked():
            command = "roslaunch auv_teleop start_teleop.launch controller:=xbox"
        else:
            command = "roslaunch auv_teleop start_teleop.launch"
        print(f"Executing: {command}")
        self.processes["teleop"] = subprocess.Popen(command, shell=True)

    def stop_teleop(self):
        if "teleop" in self.processes:
            self.processes["teleop"].terminate()
            self.processes["teleop"].wait()
            del self.processes["teleop"]
            self.output_display.append("Teleop Stopped")

        kill_joy_node = "rosnode kill /taluy/joy_node"
        kill_joystick_node = "rosnode kill /taluy/joystick_node"
        subprocess.Popen(kill_joy_node, shell=True)
        subprocess.Popen(kill_joystick_node, shell=True)
        self.output_display.append("Killed /taluy/joy_node and /taluy/joystick_node")

        reset_command = '''rostopic pub /taluy/cmd_vel geometry_msgs/Twist -1 "linear:
          x: 0.0
          y: 0.0
          z: 0.0
        angular:
          x: 0.0
          y: 0.0
          z: 0.0"'''
        subprocess.Popen(reset_command, shell=True)
        self.output_display.append("cmd_vel topic reset to zero")

    def start_detection(self):
        command = "roslaunch auv_detection tracker.launch"
        print(f"Executing: {command}")
        self.processes["detection"] = subprocess.Popen(command, shell=True)

    def stop_detection(self):
        if "detection" in self.processes:
            self.processes["detection"].terminate()
            self.processes["detection"].wait()
            del self.processes["detection"]
            self.output_display.append("Detection Stopped")

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

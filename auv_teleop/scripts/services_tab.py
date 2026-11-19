#!/usr/bin/env python3

import rclpy
from auv_common_lib.ros import service_client, get_node, ROSException
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QDoubleSpinBox,
    QHBoxLayout,
    QCheckBox,
    QLabel,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from std_msgs.msg import Bool
from std_srvs.srv import Empty, Trigger, SetBool
from auv_msgs.srv import SetDepth


class ROSServiceCaller:
    def _call_trigger_service(self, service_name):
        try:
            trigger_service = service_client(get_node(), Trigger, service_name)
            trigger_service.wait_for_service(timeout=1)
            response = trigger_service(Trigger.Request())
            if not response.success:
                print(f"{service_name} failed: {response.message}")
            return response.success
        except Exception as e:
            print(f"{service_name} call failed: {e}")
            return False
        except ROSException as e:
            print(f"{service_name} not available: {e}")
            return False

    def set_depth(self, target_depth):
        try:
            set_depth_service = service_client(get_node(), SetDepth, "set_depth")
            set_depth_service.wait_for_service(timeout=1)
            request = SetDepth.Request()
            request.target_depth = target_depth
            response = set_depth_service(request)
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def calibrate_imu(self, duration):
        try:
            from auv_msgs.srv import CalibrateIMU

            calibrate_service = service_client(
                get_node(), CalibrateIMU, "calibrate_imu"
            )
            calibrate_service.wait_for_service(timeout=1)
            request = CalibrateIMU.Request()
            request.duration = float(duration)
            response = calibrate_service(request)
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except (ROSException, ImportError) as e:
            print(f"Service not available or import error: {e}")
            return False

    def start_localization(self):
        try:
            localization_service = service_client(
                get_node(), Empty, "localization_enable"
            )
            localization_service.wait_for_service(timeout=1)
            localization_service(Empty.Request())
            return True
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def enable_dvl(self):
        try:
            dvl_service = service_client(get_node(), SetBool, "dvl/enable")
            dvl_service.wait_for_service(timeout=1)
            request = SetBool.Request()
            request.data = True
            response = dvl_service(data=True)
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def clear_objects(self):
        object_transforms_cleared = self._call_trigger_service(
            "clear_object_transforms"
        )
        kde_buffers_cleared = self._call_trigger_service("kde_map/clear")
        if object_transforms_cleared and kde_buffers_cleared:
            print("Objects and KDE buffers cleared successfully")
        return object_transforms_cleared and kde_buffers_cleared

    def cancel_alignment(self):
        try:
            cancel_alignment_service = service_client(
                get_node(), Trigger, "align_frame/cancel"
            )
            cancel_alignment_service.wait_for_service(timeout=1)
            response = cancel_alignment_service(Trigger.Request())
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def disable_dvl(self):
        try:
            dvl_service = service_client(get_node(), SetBool, "dvl/enable")
            dvl_service.wait_for_service(timeout=1)
            request = SetBool.Request()
            request.data = False
            response = dvl_service(data=False)
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def reset_pose(self):
        try:
            reset_odometry_service = service_client(
                get_node(), Trigger, "reset_odometry"
            )
            reset_odometry_service.wait_for_service(timeout=1)
            response = reset_odometry_service(Trigger.Request())
            return response.success, response.message
        except Exception as e:
            print(f"Service call failed: {e}")
            return False, str(e)
        except ROSException as e:
            print(f"Service not available: {e}")
            return False, str(e)

    def drop_ball(self):
        try:
            ball_dropper_service = service_client(
                get_node(), Trigger, "ball_dropper/drop"
            )
            ball_dropper_service.wait_for_service(timeout=1)
            response = ball_dropper_service(Trigger.Request())
            if response.success:
                print("Ball dropped successfully")
            else:
                print(f"Failed to drop ball: {response.message}")
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def launch_torpedo(self, torpedo_id):
        service_name = f"{torpedo_id}/launch"
        try:
            torpedo_service = service_client(get_node(), Trigger, service_name)
            torpedo_service.wait_for_service(timeout=1)
            response = torpedo_service(Trigger.Request())
            if response.success:
                print(f"{torpedo_id.capitalize()} launched successfully")
            else:
                print(f"Failed to launch {torpedo_id}: {response.message}")
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False

    def enable_all_detections(self, enable):
        try:
            service = service_client(
                get_node(), SetBool, "vision/enable_all_camera_detections"
            )
            service.wait_for_service(timeout=1)
            request = SetBool.Request()
            request.data = enable
            response = service(request)
            return response.success
        except Exception as e:
            print(f"Service call failed: {e}")
            return False
        except ROSException as e:
            print(f"Service not available: {e}")
            return False


class ServicesTab(QWidget):
    def __init__(self, include_simulation=True):
        super().__init__()
        self.include_simulation = include_simulation
        self.ros_service_caller = ROSServiceCaller()
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

        # IMU Calibration
        imu_calib_group = QGroupBox("IMU Calibration")
        imu_calib_layout = QHBoxLayout()
        self.imu_calib_spin = QDoubleSpinBox()
        self.imu_calib_spin.setRange(1.0, 600.0)
        self.imu_calib_spin.setValue(150.0)
        self.imu_calib_spin.setSingleStep(30.0)
        self.imu_calib_spin.setDecimals(1)
        self.imu_calib_btn = QPushButton("Calibrate IMU")
        imu_calib_layout.addWidget(QLabel("Duration:"))
        imu_calib_layout.addWidget(self.imu_calib_spin)
        imu_calib_layout.addWidget(self.imu_calib_btn)
        imu_calib_group.setLayout(imu_calib_layout)

        button_width = 120
        button_height = 75

        # Services
        service_group = QGroupBox("Services")
        service_main_layout = QHBoxLayout()

        service_left_layout = QVBoxLayout()

        # First row of buttons
        first_row = QHBoxLayout()
        self.localization_btn = QPushButton("Start\nLocalization")
        self.localization_btn.setFixedSize(button_width, button_height)
        self.dvl_btn = QPushButton("Enable\nDVL")
        self.dvl_btn.setFixedSize(button_width, button_height)
        self.clear_objects_btn = QPushButton("Clear\nObjects")
        self.clear_objects_btn.setFixedSize(button_width, button_height)
        first_row.addWidget(self.localization_btn, 0, Qt.AlignLeft)
        first_row.addStretch(1)
        first_row.addWidget(self.dvl_btn, 0, Qt.AlignLeft)
        first_row.addStretch(1)
        first_row.addWidget(self.clear_objects_btn, 0, Qt.AlignLeft)

        # Second row of buttons
        second_row = QHBoxLayout()
        self.cancel_align_btn = QPushButton("Cancel\nAlignment")
        self.cancel_align_btn.setFixedSize(button_width, button_height)
        self.disable_dvl_btn = QPushButton("Disable\nDVL")
        self.disable_dvl_btn.setFixedSize(button_width, button_height)
        self.reset_pose_btn = QPushButton("Reset\nPose")
        self.reset_pose_btn.setFixedSize(button_width, button_height)
        second_row.addWidget(self.cancel_align_btn, 0, Qt.AlignLeft)
        second_row.addStretch(1)
        second_row.addWidget(self.disable_dvl_btn, 0, Qt.AlignLeft)
        second_row.addStretch(1)
        second_row.addWidget(self.reset_pose_btn, 0, Qt.AlignLeft)

        service_left_layout.addLayout(first_row)
        service_left_layout.addLayout(second_row)

        # Right side layout for Vision Enable/Disable
        service_right_layout = QVBoxLayout()
        service_right_layout.addStretch(1)

        self.enable_all_vision_btn = QPushButton("Enable All\nVision")
        self.enable_all_vision_btn.setFixedSize(120, 60)
        self.enable_all_vision_btn.setStyleSheet("font-size: 11px;")

        self.disable_all_vision_btn = QPushButton("Disable All\nVision")
        self.disable_all_vision_btn.setFixedSize(120, 60)
        self.disable_all_vision_btn.setStyleSheet("font-size: 11px;")

        service_right_layout.addWidget(self.enable_all_vision_btn, 0, Qt.AlignCenter)
        service_right_layout.addSpacing(10)
        service_right_layout.addWidget(self.disable_all_vision_btn, 0, Qt.AlignCenter)
        service_right_layout.addStretch(1)

        service_main_layout.addLayout(service_left_layout)
        service_main_layout.addSpacing(5)
        service_main_layout.addLayout(service_right_layout)

        service_group.setLayout(service_main_layout)

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
        if not self.include_simulation:
            layout.addWidget(imu_calib_group)
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
        if not self.include_simulation:
            self.imu_calib_btn.clicked.connect(self.on_calibrate_imu)
        self.dvl_btn.clicked.connect(self.enable_dvl)
        self.clear_objects_btn.clicked.connect(self.clear_objects)
        self.cancel_align_btn.clicked.connect(self.cancel_alignment)
        self.disable_dvl_btn.clicked.connect(self.disable_dvl)
        self.reset_pose_btn.clicked.connect(self.reset_pose)
        self.torpedo1_btn.clicked.connect(lambda: self.launch_torpedo("torpedo_1"))
        self.torpedo2_btn.clicked.connect(lambda: self.launch_torpedo("torpedo_2"))
        self.enable_all_vision_btn.clicked.connect(self.enable_all_vision)
        self.disable_all_vision_btn.clicked.connect(self.disable_all_vision)

    def set_depth(self):
        depth = self.depth_spin.value()
        result = self.ros_service_caller.set_depth(depth)
        if result:
            print(f"Setting depth to: {depth}")
        else:
            QMessageBox.warning(self, "Error", "Failed to set depth.")

    def on_calibrate_imu(self):
        duration = self.imu_calib_spin.value()
        success = self.ros_service_caller.calibrate_imu(duration)
        if success:
            QMessageBox.information(
                self,
                "Success",
                f"IMU calibration finished with duration {duration:.1f}s",
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to start IMU calibration")

    def toggle_missions(self, state):
        enable = state == Qt.Checked
        self.drop_ball_btn.setEnabled(enable)
        self.torpedo1_btn.setEnabled(enable)
        self.torpedo2_btn.setEnabled(enable)

    def start_localization(self):
        result = self.ros_service_caller.start_localization()
        if result:
            print("Starting localization")
        else:
            QMessageBox.warning(self, "Error", "Failed to start localization.")

    def enable_dvl(self):
        result = self.ros_service_caller.enable_dvl()
        if result:
            print("Enabling DVL")
        else:
            QMessageBox.warning(self, "Error", "Failed to enable DVL.")

    def clear_objects(self):
        result = self.ros_service_caller.clear_objects()
        if result:
            print("Objects cleared successfully")
        else:
            QMessageBox.warning(self, "Error", "Failed to clear objects.")

    def cancel_alignment(self):
        result = self.ros_service_caller.cancel_alignment()
        if result:
            print("Alignment cancelled")
        else:
            QMessageBox.warning(self, "Error", "Failed to cancel alignment.")

    def disable_dvl(self):
        result = self.ros_service_caller.disable_dvl()
        if result:
            print("DVL disabled")
        else:
            QMessageBox.warning(self, "Error", "Failed to disable DVL.")

    def reset_pose(self):
        result, message = self.ros_service_caller.reset_pose()
        if result:
            print(message)
        else:
            QMessageBox.warning(self, "Error", message)

    def drop_ball(self):
        result = self.ros_service_caller.drop_ball()
        if not result:
            QMessageBox.warning(self, "Error", "Failed to drop ball.")

    def launch_torpedo(self, torpedo_id):
        result = self.ros_service_caller.launch_torpedo(torpedo_id)
        if not result:
            QMessageBox.warning(self, "Error", f"Failed to launch {torpedo_id}.")

    def enable_all_vision(self):
        result = self.ros_service_caller.enable_all_detections(True)
        if result:
            print("All vision/detections enabled")
        else:
            QMessageBox.warning(self, "Error", "Failed to enable all vision.")

    def disable_all_vision(self):
        result = self.ros_service_caller.enable_all_detections(False)
        if result:
            print("All vision/detections disabled")
        else:
            QMessageBox.warning(self, "Error", "Failed to disable all vision.")

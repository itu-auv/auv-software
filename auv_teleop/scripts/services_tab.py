#!/usr/bin/env python3

import rospy
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
from std_srvs.srv import (
    Empty,
    EmptyRequest,
    Trigger,
    TriggerRequest,
    SetBool,
    SetBoolRequest,
)
from auv_msgs.srv import SetDepth, SetDepthRequest


class ROSServiceCaller:
    def set_depth(self, target_depth):
        try:
            rospy.wait_for_service("set_depth", timeout=1)
            set_depth_service = rospy.ServiceProxy("set_depth", SetDepth)
            request = SetDepthRequest()
            request.target_depth = target_depth
            response = set_depth_service(request)
            return response.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        except rospy.ROSException as e:
            print(f"Service not available: {e}")
            return False

    def start_localization(self):
        try:
            rospy.wait_for_service("localization_enable", timeout=1)
            localization_service = rospy.ServiceProxy("localization_enable", Empty)
            localization_service(EmptyRequest())
            return True
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        except rospy.ROSException as e:
            print(f"Service not available: {e}")
            return False

    def enable_dvl(self):
        try:
            rospy.wait_for_service("dvl/enable", timeout=1)
            dvl_service = rospy.ServiceProxy("dvl/enable", SetBool)
            request = SetBoolRequest()
            request.data = True
            response = dvl_service(data=True)
            return response.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        except rospy.ROSException as e:
            print(f"Service not available: {e}")
            return False

    def clear_objects(self):
        try:
            rospy.wait_for_service("clear_object_transforms", timeout=1)
            clear_objects_service = rospy.ServiceProxy(
                "clear_object_transforms", Trigger
            )
            response = clear_objects_service(TriggerRequest())
            if response.success:
                print("Objects cleared successfully")
            else:
                print(f"Failed to clear objects: {response.message}")
            return response.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        except rospy.ROSException as e:
            print(f"Service not available: {e}")
            return False

    def drop_ball(self):
        try:
            rospy.wait_for_service("ball_dropper/drop", timeout=1)
            ball_dropper_service = rospy.ServiceProxy("ball_dropper/drop", Trigger)
            response = ball_dropper_service(TriggerRequest())
            if response.success:
                print("Ball dropped successfully")
            else:
                print(f"Failed to drop ball: {response.message}")
            return response.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        except rospy.ROSException as e:
            print(f"Service not available: {e}")
            return False

    def launch_torpedo(self, torpedo_id):
        service_name = f"{torpedo_id}/launch"
        try:
            rospy.wait_for_service(service_name, timeout=1)
            torpedo_service = rospy.ServiceProxy(service_name, Trigger)
            response = torpedo_service(TriggerRequest())
            if response.success:
                print(f"{torpedo_id.capitalize()} launched successfully")
            else:
                print(f"Failed to launch {torpedo_id}: {response.message}")
            return response.success
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
        except rospy.ROSException as e:
            print(f"Service not available: {e}")
            return False


class ServicesTab(QWidget):
    def __init__(self):
        super().__init__()
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

        button_width = 120
        button_height = 75

        # Services
        service_group = QGroupBox("Services")
        service_layout = QHBoxLayout()
        self.localization_btn = QPushButton("Start\nLocalization")
        self.localization_btn.setFixedSize(button_width, button_height)
        self.dvl_btn = QPushButton("Enable\nDVL")
        self.dvl_btn.setFixedSize(button_width, button_height)
        self.clear_objects_btn = QPushButton("Clear\nObjects")
        self.clear_objects_btn.setFixedSize(button_width, button_height)
        service_layout.addWidget(self.localization_btn, 0, Qt.AlignLeft)
        service_layout.addStretch(1)
        service_layout.addWidget(self.dvl_btn, 0, Qt.AlignLeft)
        service_layout.addStretch(1)
        service_layout.addWidget(self.clear_objects_btn, 0, Qt.AlignLeft)
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
        self.clear_objects_btn.clicked.connect(self.clear_objects)
        self.torpedo1_btn.clicked.connect(lambda: self.launch_torpedo("torpedo_1"))
        self.torpedo2_btn.clicked.connect(lambda: self.launch_torpedo("torpedo_2"))

    def set_depth(self):
        depth = self.depth_spin.value()
        result = self.ros_service_caller.set_depth(depth)
        if result:
            print(f"Setting depth to: {depth}")
        else:
            QMessageBox.warning(self, "Error", "Failed to set depth.")

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

    def drop_ball(self):
        result = self.ros_service_caller.drop_ball()
        if not result:
            QMessageBox.warning(self, "Error", "Failed to drop ball.")

    def launch_torpedo(self, torpedo_id):
        result = self.ros_service_caller.launch_torpedo(torpedo_id)
        if not result:
            QMessageBox.warning(self, "Error", f"Failed to launch {torpedo_id}.")

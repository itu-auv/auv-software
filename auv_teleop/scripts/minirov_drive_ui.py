#!/usr/bin/env python3

import rospy
import threading
import time
from std_msgs.msg import Bool, UInt16MultiArray
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QLabel,
    QGroupBox,
    QProgressBar,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont

STYLE = """
QMainWindow, QWidget { background-color: #1e1e1e; color: #e0e0e0; }
QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 10px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QFrame#motorCard { border-radius: 8px; border: 2px solid #555; }
QFrame#motorCard[state="forward"] { background-color: #1b5e20; border-color: #4caf50; }
QFrame#motorCard[state="reverse"] { background-color: #7f1d1d; border-color: #ef5350; }
QFrame#motorCard[state="neutral"] { background-color: #2d2d2d; }
QFrame#motorCard[state="dead"] { background-color: #3a2e00; border-color: #8a6d00; }
QProgressBar { background-color: #444; border: none; border-radius: 3px; height: 12px; }
QProgressBar::chunk { background-color: #ffd54f; border-radius: 3px; }
QLabel#motorName { font-weight: bold; color: #f5f5f5; font-size: 13px; }
QLabel#pwmLabel { font-family: monospace; font-size: 20px; color: #ffd54f; }
QLabel#dirLabel { font-size: 15px; }
QFrame#hull { background-color: #263238; border: 2px solid #546e7a; border-radius: 10px; }
"""

MOTOR_REGIONS = {
    "left_x": {"pos": (0, 0), "id": 0},
    "right_x": {"pos": (0, 2), "id": 1},
    "left_z": {"pos": (1, 0), "id": 2},
    "right_z": {"pos": (1, 2), "id": 3},
}


class MotorCard(QFrame):
    def __init__(self, name, motor_id):
        super().__init__()
        self.setObjectName("motorCard")
        self.motor_id = motor_id
        layout = QVBoxLayout(self)
        layout.setSpacing(3)
        layout.setContentsMargins(10, 8, 10, 8)

        name_label = QLabel(name)
        name_label.setObjectName("motorName")
        name_label.setAlignment(Qt.AlignCenter)

        self.pwm_label = QLabel("----")
        self.pwm_label.setObjectName("pwmLabel")
        self.pwm_label.setAlignment(Qt.AlignCenter)

        self.dir_label = QLabel("--")
        self.dir_label.setObjectName("dirLabel")
        self.dir_label.setAlignment(Qt.AlignCenter)
        self.dir_label.setStyleSheet("font-size: 16px;")

        self.bar = QProgressBar()
        self.bar.setRange(1200, 1800)
        self.bar.setTextVisible(False)

        layout.addWidget(name_label)
        layout.addWidget(self.pwm_label)
        layout.addWidget(self.dir_label)
        layout.addWidget(self.bar)

    def update_pwm(self, pwm):
        if pwm is None:
            self.pwm_label.setText("----")
            self.dir_label.setText("--")
            self.setProperty("state", "dead")
            self.bar.setValue(1500)
        else:
            self.pwm_label.setText(str(pwm))
            if pwm > 1500:
                self.dir_label.setText("^" if self.motor_id >= 2 else ">")
                self.setProperty("state", "forward")
            elif pwm < 1500:
                self.dir_label.setText("v" if self.motor_id >= 2 else "<")
                self.setProperty("state", "reverse")
            else:
                self.dir_label.setText("o")
                self.setProperty("state", "neutral")
            self.bar.setValue(pwm)
        self.style().unpolish(self)
        self.style().polish(self)


class MiniRovDriveUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniROV Drive Pulse UI")
        self.resize(640, 420)

        self.topic = rospy.get_param("~pulse_topic", "/minirov/drive_pulse")
        self.lumen_topic = rospy.get_param(
            "~lumen_topic", "/minirov/lumen_brightness"
        )

        self.lock = threading.Lock()
        self.pwm_data = None
        self.lumen_enabled = None
        self.last_msg_time = 0.0
        self.msg_count = 0
        self.msg_rate = 0.0

        self._build_ui()

        rospy.Subscriber(
            self.topic, UInt16MultiArray, self.pulse_callback, queue_size=1
        )
        rospy.Subscriber(
            self.lumen_topic, Bool, self.lumen_callback, queue_size=1
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(50)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.setStyleSheet(STYLE)

        root = QVBoxLayout(central)

        self.status_label = QLabel(f"Waiting for data on {self.topic} ...")
        self.status_label.setFont(QFont("monospace", 10))
        self.status_label.setStyleSheet("color: #ffb74d;")
        root.addWidget(self.status_label)

        grid = QGridLayout()
        grid.setSpacing(12)
        root.addLayout(grid, 1)

        self.motor_cards = {}
        for motor_id in sorted(set(p["id"] for p in MOTOR_REGIONS.values())):
            name = [n for n, p in MOTOR_REGIONS.items() if p["id"] == motor_id][0]
            card = MotorCard(name, motor_id)
            self.motor_cards[motor_id] = card

        for name, region in MOTOR_REGIONS.items():
            row, col = region["pos"]
            grid.addWidget(self.motor_cards[region["id"]], row, col)

        hull = QFrame()
        hull.setObjectName("hull")
        hull_layout = QVBoxLayout(hull)
        hull_layout.setAlignment(Qt.AlignCenter)
        hull_title = QLabel("MiniROV")
        hull_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #b0bec5;")
        hull_title.setAlignment(Qt.AlignCenter)
        hull_layout.addWidget(hull_title)

        self.raw_label = QLabel("raw: []")
        self.raw_label.setStyleSheet("font-family: monospace; color: #90a4ae;")
        self.raw_label.setAlignment(Qt.AlignCenter)
        hull_layout.addWidget(self.raw_label)

        self.lumen_label = QLabel("Lumen: waiting")
        self.lumen_label.setAlignment(Qt.AlignCenter)
        hull_layout.addWidget(self.lumen_label)

        hull_layout.addStretch()
        grid.addWidget(hull, 0, 1, 2, 1)

        legend = QLabel(
            "green = forward (>1500)   red = reverse (<1500)   yellow bar = pwm"
        )
        legend.setStyleSheet("color: #888; font-size: 10px;")
        root.addWidget(legend)

    def pulse_callback(self, msg):
        with self.lock:
            self.pwm_data = list(msg.data)
            self.last_msg_time = time.time()
            self.msg_count += 1

    def lumen_callback(self, msg):
        with self.lock:
            self.lumen_enabled = msg.data

    def update_ui(self):
        with self.lock:
            has_data = self.pwm_data is not None
            age = time.time() - self.last_msg_time if has_data else 99.0

            color = "#4caf50" if has_data else "#ffb74d"
            self.status_label.setStyleSheet(f"color: {color};")
            self.status_label.setText(
                f"Topic: {self.topic} | Connected: YES | "
                f"messages: {self.msg_count} | last: {age:.1f}s ago"
            )

            placeholder = [1500, 1500, 1500, 1500]
            values = self.pwm_data if has_data else placeholder

            for motor_id, card in self.motor_cards.items():
                if motor_id < len(values):
                    card.update_pwm(values[motor_id])
                else:
                    card.update_pwm(None)

            self.raw_label.setText("raw: " + str(values))
            if self.lumen_enabled is not None:
                value = "TRUE" if self.lumen_enabled else "FALSE"
                color = "#4caf50" if self.lumen_enabled else "#ef5350"
                self.lumen_label.setText(f"Lumen: {value}")
                self.lumen_label.setStyleSheet(
                    f"font-weight: bold; color: {color};"
                )


if __name__ == "__main__":
    import signal
    import sys

    app = QApplication(sys.argv)
    rospy.init_node("minirov_drive_ui")

    def shutdown_ui(_signal=None, _frame=None):
        rospy.signal_shutdown("UI closed")
        app.quit()

    signal.signal(signal.SIGINT, shutdown_ui)
    rospy.on_shutdown(app.quit)

    window = MiniRovDriveUI()
    window.show()
    app.exec_()

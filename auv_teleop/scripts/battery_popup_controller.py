#!/usr/bin/env python3

import random
from collections import deque
from typing import Callable, Optional

import rospy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sensor_msgs.msg import BatteryState


FUNNY_MESSAGES = [
    "Colorado olacağız",
    "Abi biz onu beceremeyiz",
    "Abi bir deneyelim belki pillere bir şey olmaz?",
    "Abi yemek siparişini pillerin bitişine denk getirin",
    "Oh man",
    "Minoso parlare italiano",
    "Biraz da taluy miniyi test edin",
]


POPUP_STYLE = """
QWidget#popup {
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 #1a0000, stop:0.5 #330000, stop:1 #1a0000
    );
    border: 4px solid #ff2222;
    border-radius: 24px;
}
"""

TITLE_STYLE = """
    color: #ff4444;
    font-size: 52px;
    font-weight: bold;
    padding: 10px;
"""

VOLTAGE_STYLE = """
    color: #ff6666;
    font-size: 72px;
    font-weight: bold;
    padding: 5px;
"""

FUNNY_STYLE = """
    color: #ffaa66;
    font-size: 26px;
    font-style: italic;
    padding: 20px 40px;
"""

DISMISS_STYLE = """
    QPushButton {
        background-color: #cc0000;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 12px 48px;
    }
    QPushButton:hover {
        background-color: #ff2222;
    }
    QPushButton:pressed {
        background-color: #990000;
    }
"""


class BatteryPopup(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("popup")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setStyleSheet(POPUP_STYLE)
        self.dismiss_time = rospy.Time(0)
        self.is_alert_visible = False
        self.on_dismiss: Optional[Callable[[], None]] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(10)

        self.title_label = QLabel("LOW BATTERY")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(TITLE_STYLE)
        self._add_glow(self.title_label, "#ff0000", 30)

        self.voltage_label = QLabel("0.00 V")
        self.voltage_label.setAlignment(Qt.AlignCenter)
        self.voltage_label.setStyleSheet(VOLTAGE_STYLE)
        self._add_glow(self.voltage_label, "#ff4444", 20)

        self.threshold_label = QLabel("")
        self.threshold_label.setAlignment(Qt.AlignCenter)
        self.threshold_label.setStyleSheet("color: #aa4444; font-size: 18px;")

        self.funny_label = QLabel("")
        self.funny_label.setAlignment(Qt.AlignCenter)
        self.funny_label.setWordWrap(True)
        self.funny_label.setStyleSheet(FUNNY_STYLE)

        dismiss_btn = QPushButton("DISMISS")
        dismiss_btn.setStyleSheet(DISMISS_STYLE)
        dismiss_btn.setCursor(Qt.PointingHandCursor)
        dismiss_btn.clicked.connect(self._on_dismiss)

        layout.addStretch(1)
        layout.addWidget(self.title_label)
        layout.addWidget(self.voltage_label)
        layout.addWidget(self.threshold_label)
        layout.addSpacing(10)
        layout.addWidget(self.funny_label)
        layout.addStretch(1)
        layout.addWidget(dismiss_btn, alignment=Qt.AlignCenter)
        layout.addSpacing(20)

    def _on_dismiss(self):
        self.dismiss_time = rospy.Time.now()
        self.is_alert_visible = False
        self.hide()
        if self.on_dismiss is not None:
            self.on_dismiss()

    def closeEvent(self, event):
        self._on_dismiss()
        event.accept()

    def show_alert(self, voltage: float, threshold: float):
        self.voltage_label.setText(f"{voltage:.2f} V")
        self.threshold_label.setText(f"Threshold: {threshold:.1f} V")
        self.funny_label.setText(random.choice(FUNNY_MESSAGES))

        screen = QApplication.primaryScreen().geometry()
        width = min(800, screen.width() - 100)
        height = min(500, screen.height() - 100)
        x_pos = (screen.width() - width) // 2
        y_pos = (screen.height() - height) // 2
        self.setGeometry(x_pos, y_pos, width, height)

        self.is_alert_visible = True
        self.show()
        self.raise_()
        self.activateWindow()

    @staticmethod
    def _add_glow(widget: QWidget, color: str, radius: int):
        effect = QGraphicsDropShadowEffect()
        effect.setBlurRadius(radius)
        effect.setColor(QColor(color))
        effect.setOffset(0, 0)
        widget.setGraphicsEffect(effect)


class BatteryPopupController:
    def __init__(
        self,
        topic: str = "power",
        voltage_threshold: float = 13.5,
        window_size: int = 5,
        cooldown_seconds: float = 180.0,
    ):
        self.voltage_threshold = voltage_threshold
        self.window_size = max(1, int(window_size))
        self.cooldown_seconds = float(cooldown_seconds)
        self.readings = deque(maxlen=self.window_size)
        self.pending_voltage = None
        self.popup_active = False

        self.popup = BatteryPopup()
        self.popup.on_dismiss = self._on_popup_dismiss
        self.subscriber = rospy.Subscriber(topic, BatteryState, self._battery_cb)

        self.qt_timer = QTimer()
        self.qt_timer.timeout.connect(self._main_thread_tick)
        self.qt_timer.start(100)

        rospy.loginfo(
            "Battery popup active | topic=%s threshold=%.2fV window=%d cooldown=%.1fs",
            topic,
            self.voltage_threshold,
            self.window_size,
            self.cooldown_seconds,
        )

    def _battery_cb(self, msg: BatteryState):
        self.readings.append(msg.voltage)

        if len(self.readings) < self.window_size:
            return

        if not all(v < self.voltage_threshold for v in self.readings):
            return

        if self.pending_voltage is not None or self.popup_active:
            return

        now = rospy.Time.now()
        since_dismiss = (now - self.popup.dismiss_time).to_sec()
        if since_dismiss < self.cooldown_seconds:
            return

        self.popup_active = True
        self.pending_voltage = msg.voltage
        rospy.logwarn(
            "Battery critically low: %.2fV (all last %d readings below %.1fV)",
            msg.voltage,
            self.window_size,
            self.voltage_threshold,
        )

    def _on_popup_dismiss(self):
        self.popup_active = False

    def _main_thread_tick(self):
        if self.pending_voltage is None:
            return

        voltage = self.pending_voltage
        self.pending_voltage = None
        self.popup.show_alert(voltage, self.voltage_threshold)

    def shutdown(self):
        self.qt_timer.stop()
        self.subscriber.unregister()

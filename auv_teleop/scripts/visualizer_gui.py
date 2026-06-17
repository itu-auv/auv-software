#!/usr/bin/env python3

import sys
import os
import threading
import rospy
import tf
import math
import yaml
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QGridLayout,
    QLabel,
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QListWidget,
    QDialogButtonBox,
)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QTransform
from PyQt5.QtCore import Qt, QTimer
from sensor_msgs.msg import CompressedImage

CONFIG_FILE = "visiuliser.yaml"
IMAGE_DISPLAY_INTERVAL_MS = 50
IMAGE_SUBSCRIBER_BUFF_SIZE = 2**24

class MapWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.tf_listener = tf.TransformListener()
        self.setStyleSheet("background-color: black;")
        
        # history for blue dots
        self.history = []
        
        # timers
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_map)
        self.update_timer.start(50)  # 20 Hz
        
        self.history_timer = QTimer(self)
        self.history_timer.timeout.connect(self.save_history)
        self.history_timer.start(15000)  # 15 seconds
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.has_transform = False

    def update_map(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('odom', 'taluy/base_link', rospy.Time(0))
            self.current_x = trans[0]
            self.current_y = trans[1]
            euler = tf.transformations.euler_from_quaternion(rot)
            self.current_yaw = euler[2]
            self.has_transform = True
            self.update()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass

    def save_history(self):
        if self.has_transform:
            self.history.append((self.current_x, self.current_y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # 40x40 resolution (viewport is 40m x 40m)
        scale_x = width / 40.0
        scale_y = height / 40.0
        
        # Center coordinates
        cx = width / 2.0
        cy = height / 2.0
        
        def to_pixel(x, y):
            # x is forward, y is left in ROS.
            # In GUI, x is right, y is down.
            # Let's map ROS x to GUI up (-y), ROS y to GUI left (-x)
            px = cx - y * scale_x
            py = cy - x * scale_y
            return px, py

        # Draw grid
        pen_grid = QPen(QColor(50, 50, 50))
        pen_grid.setWidth(1)
        painter.setPen(pen_grid)
        
        # Grid lines every 1 meter from -20 to 20 (for a 40x40 map)
        for i in range(-20, 21):
            px1, py1 = to_pixel(-20, i)
            px2, py2 = to_pixel(20, i)
            painter.drawLine(int(px1), int(py1), int(px2), int(py2))
            
            px1, py1 = to_pixel(i, -20)
            px2, py2 = to_pixel(i, 20)
            painter.drawLine(int(px1), int(py1), int(px2), int(py2))

        # Draw history (blue dots)
        painter.setBrush(QBrush(QColor("blue")))
        painter.setPen(Qt.NoPen)
        for hx, hy in self.history:
            px, py = to_pixel(hx, hy)
            painter.drawEllipse(int(px) - 3, int(py) - 3, 6, 6)

        # Draw odom (green dot)
        painter.setBrush(QBrush(QColor("green")))
        px, py = to_pixel(0, 0)
        painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)
        
        if self.has_transform:
            # Draw base_link (red dot)
            painter.setBrush(QBrush(QColor("red")))
            px, py = to_pixel(self.current_x, self.current_y)
            painter.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)
            
            # Draw arrow
            arrow_len = 1.0 # 1 meter long arrow
            end_x = self.current_x + math.cos(self.current_yaw) * arrow_len
            end_y = self.current_y + math.sin(self.current_yaw) * arrow_len
            
            ex, ey = to_pixel(end_x, end_y)
            
            pen = QPen(QColor("red"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(int(px), int(py), int(ex), int(ey))

def load_visualizer_configs(config_path):
    with open(config_path, "r") as config_file:
        configs = yaml.safe_load(config_file)

    if not isinstance(configs, dict) or not configs:
        raise ValueError("Visualizer config must contain at least one mode.")

    for mode_name, topics in configs.items():
        if not isinstance(topics, list) or len(topics) != 3:
            raise ValueError("Mode '{}' must contain exactly 3 topics.".format(mode_name))

        for topic in topics:
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError("Mode '{}' contains an empty topic.".format(mode_name))

    return configs


def normalize_topic(topic):
    topic = topic.strip()
    if not topic.startswith("/"):
        topic = "/" + topic
    if any(char.isspace() for char in topic):
        raise ValueError("Topic '{}' contains whitespace.".format(topic))
    return topic


class ConfigSelectionDialog(QDialog):
    def __init__(self, mode_names):
        super().__init__()
        self.setWindowTitle("Select Visualizer Config")
        self.resize(560, 420)

        layout = QVBoxLayout(self)

        title = QLabel("Choose visualizer mode")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 12px;")
        layout.addWidget(title)

        self.mode_list = QListWidget()
        self.mode_list.addItems(mode_names)
        self.mode_list.setCurrentRow(0)
        self.mode_list.setStyleSheet("font-size: 22px; padding: 10px;")
        self.mode_list.itemDoubleClicked.connect(self.accept)
        layout.addWidget(self.mode_list)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_mode(self):
        selected_item = self.mode_list.currentItem()
        if selected_item is None:
            return None
        return selected_item.text()


def choose_visualizer_config(configs):
    mode_names = list(configs.keys())
    dialog = ConfigSelectionDialog(mode_names)

    if dialog.exec_() != QDialog.Accepted:
        return None, None

    selected_mode = dialog.selected_mode()
    return selected_mode, [normalize_topic(topic) for topic in configs[selected_mode]]


class VisualizerGUI(QWidget):
    def __init__(self, mode_name, topics):
        super().__init__()
        self.setWindowTitle("Camera & Map Visualizer - {}".format(mode_name))
        self.resize(800, 600)
        
        layout = QGridLayout()
        self.setLayout(layout)
        
        self.label_front = QLabel()
        self.label_front.setAlignment(Qt.AlignCenter)
        self.label_front.setStyleSheet("background-color: #222; color: white;")
        
        self.label_bottom = QLabel()
        self.label_bottom.setAlignment(Qt.AlignCenter)
        self.label_bottom.setStyleSheet("background-color: #222; color: white;")
        
        self.label_torpedo = QLabel()
        self.label_torpedo.setAlignment(Qt.AlignCenter)
        self.label_torpedo.setStyleSheet("background-color: #222; color: white;")

        self.set_label_texts(topics)
        
        self.map_widget = MapWidget()
        
        layout.addWidget(self.label_front, 0, 0)
        layout.addWidget(self.label_torpedo, 0, 1)
        layout.addWidget(self.label_bottom, 1, 0)
        layout.addWidget(self.map_widget, 1, 1)

        self._frame_lock = threading.Lock()
        self._latest_frames = {
            "front": None,
            "bottom": None,
            "torpedo": None,
        }
        self._displayed_frame_ids = {
            "front": None,
            "bottom": None,
            "torpedo": None,
        }
        self._next_frame_id = 0
        self._streams = {
            "front": (self.label_front, 0),
            "bottom": (self.label_bottom, 270),
            "torpedo": (self.label_torpedo, 0),
        }
        
        # Subscribers come from the selected YAML mode in this order.
        self.subscribers = [
            self._subscribe_image(topics[0], self.front_cb),
            self._subscribe_image(topics[1], self.bottom_cb),
            self._subscribe_image(topics[2], self.torpedo_cb),
        ]

        self.image_timer = QTimer(self)
        self.image_timer.timeout.connect(self.update_images)
        self.image_timer.start(IMAGE_DISPLAY_INTERVAL_MS)

    def set_label_texts(self, topics):
        self.label_front.setText("Stream 1\n{}".format(topics[0]))
        self.label_bottom.setText("Stream 2\n{}".format(topics[1]))
        self.label_torpedo.setText("Stream 3\n{}".format(topics[2]))

    def _subscribe_image(self, topic, callback):
        return rospy.Subscriber(
            topic,
            CompressedImage,
            callback,
            queue_size=1,
            buff_size=IMAGE_SUBSCRIBER_BUFF_SIZE,
        )

    def _store_latest_frame(self, stream_name, msg):
        with self._frame_lock:
            self._next_frame_id += 1
            self._latest_frames[stream_name] = (self._next_frame_id, bytes(msg.data))

    def front_cb(self, msg):
        self._store_latest_frame("front", msg)

    def bottom_cb(self, msg):
        self._store_latest_frame("bottom", msg)

    def torpedo_cb(self, msg):
        self._store_latest_frame("torpedo", msg)

    def update_images(self):
        with self._frame_lock:
            frames = dict(self._latest_frames)

        for stream_name, frame in frames.items():
            if frame is None:
                continue

            frame_id, image_data = frame
            if self._displayed_frame_ids[stream_name] == frame_id:
                continue

            label, rotate = self._streams[stream_name]
            if self.update_image(label, image_data, rotate=rotate):
                self._displayed_frame_ids[stream_name] = frame_id

    def update_image(self, label, image_data, rotate=0):
        pixmap = QPixmap()
        pixmap.loadFromData(image_data)
        if pixmap.isNull():
            return False

        if rotate != 0:
            transform = QTransform().rotate(rotate)
            pixmap = pixmap.transformed(transform, Qt.FastTransformation)

        label.setPixmap(
            pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        )
        return True

if __name__ == "__main__":
    import signal

    def signal_handler(sig, frame):
        print("\nClosing GUI...")
        QApplication.quit()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.init_node("visualizer_gui")
    app = QApplication(sys.argv)

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
    try:
        configs = load_visualizer_configs(config_path)
        selected_mode, topics = choose_visualizer_config(configs)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        QMessageBox.critical(None, "Visualizer Config Error", str(exc))
        sys.exit(1)

    if selected_mode is None:
        sys.exit(0)

    gui = VisualizerGUI(selected_mode, topics)
    gui.show()

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)

    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass

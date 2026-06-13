#!/usr/bin/env python3

import sys
import rospy
import tf
import math
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen, QBrush
from PyQt5.QtCore import Qt, QTimer
from sensor_msgs.msg import CompressedImage

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
        
        # 20x20 resolution (viewport is 20m x 20m)
        scale_x = width / 20.0
        scale_y = height / 20.0
        
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

class VisualizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera & Map Visualizer")
        self.resize(800, 600)
        
        layout = QGridLayout()
        self.setLayout(layout)
        
        self.label_front = QLabel("Front Cam")
        self.label_front.setAlignment(Qt.AlignCenter)
        self.label_front.setStyleSheet("background-color: #222; color: white;")
        
        self.label_bottom = QLabel("Bottom Cam")
        self.label_bottom.setAlignment(Qt.AlignCenter)
        self.label_bottom.setStyleSheet("background-color: #222; color: white;")
        
        self.label_torpedo = QLabel("Torpedo Cam")
        self.label_torpedo.setAlignment(Qt.AlignCenter)
        self.label_torpedo.setStyleSheet("background-color: #222; color: white;")
        
        self.map_widget = MapWidget()
        
        layout.addWidget(self.label_front, 0, 0)
        layout.addWidget(self.label_bottom, 0, 1)
        layout.addWidget(self.label_torpedo, 1, 0)
        layout.addWidget(self.map_widget, 1, 1)
        
        # Subscribers
        rospy.Subscriber("/taluy/cameras/cam_front/image_rect_color/compressed", CompressedImage, self.front_cb)
        rospy.Subscriber("/taluy/cameras/cam_bottom/image_rect_color/compressed", CompressedImage, self.bottom_cb)
        rospy.Subscriber("/taluy/cameras/cam_torpedo/image_rect_color/compressed", CompressedImage, self.torpedo_cb)

    def front_cb(self, msg):
        self.update_image(self.label_front, msg)

    def bottom_cb(self, msg):
        self.update_image(self.label_bottom, msg)

    def torpedo_cb(self, msg):
        self.update_image(self.label_torpedo, msg)

    def update_image(self, label, msg):
        pixmap = QPixmap()
        pixmap.loadFromData(msg.data)
        if not pixmap.isNull():
            # scale to fit the label keeping aspect ratio
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    rospy.init_node("visualizer_gui")
    app = QApplication(sys.argv)
    gui = VisualizerGUI()
    gui.show()
    sys.exit(app.exec_())

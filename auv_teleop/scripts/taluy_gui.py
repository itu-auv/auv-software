#!/usr/bin/env python3

import sys
import os
import rospy
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QMainWindow,
    QSizePolicy,
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from dry_test_tab import DryTestTab
from services_tab import ServicesTab
from vehicle_control_tab import VehicleControlTab
from simulation_tab import SimulationTab


class MainControlPanel(QWidget):
    def __init__(self, include_simulation=True):
        super().__init__()
        self.setWindowTitle("Taluy Control Panel")

        screen = QApplication.primaryScreen()
        screen_width = screen.size().width()
        screen_height = screen.size().height()
        min_width = 400

        self.setGeometry(screen_width - min_width, 0, min_width, screen_height)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        tabs_layout = QVBoxLayout()

        if include_simulation:
            tabs_layout.addWidget(ServicesTab())
            tabs_layout.addWidget(VehicleControlTab())
            tabs_layout.addWidget(SimulationTab())
        else:
            tabs_layout.addWidget(ServicesTab())
            tabs_layout.addWidget(DryTestTab())
            tabs_layout.addWidget(VehicleControlTab())

        main_layout.addLayout(tabs_layout)
        main_layout.addStretch(0)

        self.setLayout(main_layout)


class StartScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUV GUI")

        screen = QApplication.primaryScreen()
        screen_width = screen.size().width()
        screen_height = screen.size().height()
        window_width = 1080
        window_height = 500
        x_pos = (screen_width - window_width) // 2
        y_pos = (screen_height - window_height) // 2
        self.setGeometry(x_pos, y_pos, window_width, window_height)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        central_widget.setStyleSheet("background-color: black;")

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        title_label = QLabel("ITU AUV Control Panel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 30, QFont.Bold))
        title_label.setStyleSheet(
            "color: white; background-color: black; padding: 15px;"
        )
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        button_layout.setSpacing(50)

        pool_button = QPushButton("Pool")
        pool_button.setFont(QFont("Arial", 14))
        pool_button.setMinimumSize(150, 40)
        pool_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(240, 240, 240, 220);
                color: black;
                border-radius: 5px;
                border: 1px solid #888;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(220, 220, 220, 240);
            }
        """
        )
        pool_button.clicked.connect(self.open_pool_mode)

        simulation_button = QPushButton("Simulation")
        simulation_button.setFont(QFont("Arial", 14))
        simulation_button.setMinimumSize(150, 40)
        simulation_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(240, 240, 240, 220);
                color: black;
                border-radius: 5px;
                border: 1px solid #888;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(220, 220, 220, 240);
            }
        """
        )
        simulation_button.clicked.connect(self.open_simulation_mode)

        button_layout.addWidget(pool_button)
        button_layout.addWidget(simulation_button)

        image_container = QWidget()
        image_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)
        image_path = os.path.join(package_dir, "images", "itu_auv_workshop.jpg")

        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaledToWidth(1080, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                image_label.setScaledContents(False)
            else:
                print(f"Failed to load image: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")

        image_layout.addWidget(image_label)

        main_layout.addWidget(title_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(image_container)

    def open_pool_mode(self):
        self.hide()
        self.pool_window = MainControlPanel(include_simulation=False)
        self.pool_window.show()

    def open_simulation_mode(self):
        self.hide()
        self.simulation_window = MainControlPanel(include_simulation=True)
        self.simulation_window.show()


if __name__ == "__main__":
    import signal
    
    def signal_handler(sig, frame):
        print("\nClosing GUI...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    rospy.init_node("taluy_gui_node", anonymous=True)
    app = QApplication(sys.argv)
    window = StartScreen()
    window.show()
    
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(100)
    
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass

#!/usr/bin/env python3

import sys
import os
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
from PyQt5.QtCore import Qt, QSize
from dry_test_tab import DryTestTab
from services_tab import ServicesTab
from vehicle_control_tab import VehicleControlTab
from simulation_tab import SimulationTab


class MainControlPanel(QWidget):
    def __init__(self, include_simulation=True):
        super().__init__()
        self.setWindowTitle("Taluy Control Panel")
        self.setGeometry(100, 100, 1000, 500)
        self.setMinimumHeight(200)  # Allow vertical shrinking to a minimum

        # Main layout (vertical)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)  # No margins
        main_layout.setSpacing(0)  # No spacing between elements

        # Horizontal layout for tabs
        tabs_layout = QHBoxLayout()

        # Add tabs side by side
        tabs_layout.addWidget(ServicesTab())
        tabs_layout.addWidget(DryTestTab())
        tabs_layout.addWidget(VehicleControlTab())

        # Add simulation tab conditionally
        if include_simulation:
            tabs_layout.addWidget(SimulationTab())

        # Add tabs layout to main layout
        main_layout.addLayout(tabs_layout)
        main_layout.addStretch(0)  # Prevent excessive expansion

        # Set the main layout
        self.setLayout(main_layout)


class StartScreen(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUV GUI")
        self.setGeometry(100, 100, 1080, 500)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Set black background for the entire window
        central_widget.setStyleSheet("background-color: black;")

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
        main_layout.setSpacing(5)  # Small spacing between elements

        # Title label at the top
        title_label = QLabel("ITU AUV Control Panel")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 30, QFont.Bold))
        title_label.setStyleSheet(
            "color: white; background-color: black; padding: 15px;"
        )
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Button layout - centered horizontally
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignCenter)
        button_layout.setSpacing(50)  # Horizontal spacing between buttons

        # Pool button
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

        # Simulation button
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

        # Add buttons to button layout
        button_layout.addWidget(pool_button)
        button_layout.addWidget(simulation_button)

        # Image container at the bottom
        image_container = QWidget()
        image_container.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )  # Minimum height for image
        image_layout = QVBoxLayout(image_container)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(0)

        # Create image label
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignBottom | Qt.AlignHCenter)
        image_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Minimum
        )  # Minimum height

        # Get the path to the image
        script_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(script_dir)
        image_path = os.path.join(package_dir, "images", "auv_atolye.jpg")

        try:
            # Load the image
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # Scale image to fit width while maintaining aspect ratio
                scaled_pixmap = pixmap.scaledToWidth(1080, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                image_label.setScaledContents(False)
            else:
                print(f"Failed to load image: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")

        image_layout.addWidget(image_label)

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(image_container)  # Image directly below buttons

    def open_pool_mode(self):
        self.hide()
        self.pool_window = MainControlPanel(include_simulation=False)
        self.pool_window.show()

    def open_simulation_mode(self):
        self.hide()
        self.simulation_window = MainControlPanel(include_simulation=True)
        self.simulation_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StartScreen()
    window.show()
    sys.exit(app.exec_())

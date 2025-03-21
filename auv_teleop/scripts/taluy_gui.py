#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from dry_test_tab import DryTestTab
from services_tab import ServicesTab
from vehicle_control_tab import VehicleControlTab
from simulation_tab import SimulationTab

class AUVControlGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taluy AUV Control Panel")
        self.setGeometry(100, 100, 1000, 500)  # Increased width to accommodate side-by-side tabs
        
        # Main layout (vertical)
        main_layout = QVBoxLayout()
        
        # Horizontal layout for tabs
        tabs_layout = QHBoxLayout()
        
        # Add tabs side by side
        tabs_layout.addWidget(ServicesTab())
        tabs_layout.addWidget(DryTestTab())
        tabs_layout.addWidget(VehicleControlTab())
        tabs_layout.addWidget(SimulationTab())
        
        # Add tabs layout to main layout
        main_layout.addLayout(tabs_layout)
        
        # Set the main layout
        self.setLayout(main_layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AUVControlGUI()
    window.show()
    sys.exit(app.exec_())

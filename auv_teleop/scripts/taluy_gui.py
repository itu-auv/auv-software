#!/usr/bin/env python3

import sys
from PyQt5.QtWidgets import QApplication, QTabWidget
from dry_test_tab import DryTestTab
from services_tab import ServicesTab
from vehicle_control_tab import VehicleControlTab
from simulation_tab import SimulationTab


class AUVControlGUI(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Taluy AUV Control Panel")
        self.setGeometry(100, 100, 450, 500)

        self.addTab(ServicesTab(), "Services")
        self.addTab(DryTestTab(), "Dry Test")
        self.addTab(VehicleControlTab(), "Vehicle Control")
        self.addTab(SimulationTab(), "Simulation")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AUVControlGUI()
    window.show()
    sys.exit(app.exec_())

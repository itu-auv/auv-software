#!/usr/bin/env python3

from PyQt5.QtCore import QThread, pyqtSignal
import subprocess


class CommandThread(QThread):
    output_signal = pyqtSignal(str)

    def __init__(self, command):
        super().__init__()
        self.command = command
        self._is_running = True

    def run(self):
        process = subprocess.Popen(
            self.command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        while self._is_running:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                self.output_signal.emit(output.strip())
        process.stdout.close()
        process.wait()

    def stop(self):
        self._is_running = False
#!/usr/bin/env python3

import pyrealsense2 as rs
import time

def run_self_calibration(device):
    # Auto-calibration API'sini alın
    auto_calib = device.as_auto_calibrated_device()

    # Kalibrasyonu başlatın
    auto_calib.run_on_chip_calibration()

    # Kalibrasyonun tamamlanmasını bekleyin
    time.sleep(5)

    # Kalibrasyonun durumunu kontrol edin
    health_check = auto_calib.get_calibration_health()
    if health_check == rs.auto_calibrated_device.calibration_health.ok:
        print("Kalibrasyon başarılı!")
    else:
        print("Kalibrasyon başarısız!")

def main():
    # RealSense pipeline'ını başlatın
    pipeline = rs.pipeline()

    # Konfigürasyonu oluşturun
    config = rs.config()

    # Derinlik ve renk akışlarını etkinleştirin
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    # Pipeline'ı başlatın
    profile = pipeline.start(config)

    # Cihazınızı alın
    device = profile.get_device()

    # Kendi kendine kalibrasyonu çalıştırın
    run_self_calibration(device)

    # Pipeline'ı durdurun
    pipeline.stop()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import pyrealsense2 as rs
import time

def run_self_calibration(device):
    # Auto-calibration API'sini alıyoruz.
    auto_calib = device.as_auto_calibrated_device()
    
    # On-chip kalibrasyonu başlatıyoruz. Bu işlem başarılı olursa,
    # cihazın çipine yeni kalibrasyon değerleri yazılır.
    auto_calib.run_on_chip_calibration()
    
    # Kalibrasyon işleminin tamamlanması için bekleyelim.
    time.sleep(5)
    
    # Kalibrasyon sağlığını kontrol ediyoruz.
    health_check = auto_calib.get_calibration_health()
    if health_check == rs.auto_calibrated_device.calibration_health.ok:
        print("Kalibrasyon başarılı! Yeni değerler çipe yazıldı.")
    else:
        print("Kalibrasyon başarısız!")

def read_calibration_table(device):
    # Cihazdaki sensörlerden kalibrasyon verilerini okuyup ekrana basalım.
    sensors = device.query_sensors()
    for sensor in sensors:
        try:
            # calibration_table değeri sensöre özel yeni kalibrasyon verilerini içerir.
            calibration_data = sensor.get_option(rs.option.calibration_table)
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"\nSensör: {sensor_name}")
            print("Yeni Kalibrasyon Verileri:")
            print(calibration_data)
        except Exception as e:
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"\nSensör: {sensor_name}")
            print("Bu sensör için kalibrasyon verisi okunamıyor.")
            print("Hata:", e)

def main():
    # RealSense pipeline'ını başlatıyoruz.
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Derinlik ve renk akışlarını etkinleştiriyoruz.
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)
    
    # Pipeline'ı başlatıp cihazı elde ediyoruz.
    profile = pipeline.start(config)
    device = profile.get_device()

    # On-chip kalibrasyonu çalıştırıyoruz (yeni değerler çipe yazılıyor).
    run_self_calibration(device)
    
    # Sonrasında yeni kalibrasyon değerlerini okuyup gösteriyoruz.
    read_calibration_table(device)
    
    # Pipeline'ı durduruyoruz.
    pipeline.stop()

if __name__ == "__main__":
    main()

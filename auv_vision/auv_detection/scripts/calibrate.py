#!/usr/bin/env python3

import pyrealsense2 as rs
import time

def run_self_calibration(device):
    # Auto-calibration API'sini alıyoruz.
    auto_calib = rs.auto_calibrated_device(device)
    
    # On-chip kalibrasyonu başlatıyoruz
    try:
        print("Kalibrasyon başlatılıyor...")
        print("Lütfen kamerayı düz bir yüzeye doğrultun ve sabit tutun.")
        result, health = auto_calib.run_on_chip_calibration("{}", 15000)  # 15 saniye timeout
        
        # Kalibrasyon metriklerini yazdırıyoruz
        print("\nKalibrasyon Metrikleri:")
        print(f"Kalibrasyon durumu: {'Başarılı' if result else 'Başarısız'}")
        print(f"Pyguide değeri: {health[0]:.3f}")
        print(f"RMSE değeri: {health[1]:.3f}")
        
    except Exception as e:
        print(f"Kalibrasyon sırasında hata oluştu: {e}")
    
    time.sleep(5)

def read_calibration_table(device):
    # Cihazdaki sensörlerden kalibrasyon verilerini okuyup ekrana basalım.
    sensors = device.query_sensors()
    for sensor in sensors:
        try:
            sensor_name = sensor.get_info(rs.camera_info.name)
            print(f"\nSensör: {sensor_name}")
            print("Kalibrasyon Bilgileri:")
            print(f"Seri No: {sensor.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware Versiyonu: {sensor.get_info(rs.camera_info.firmware_version)}")
            
            # Derinlik sensörü için özel bilgiler
            if "Stereo Module" in sensor_name:
                try:
                    depth_scale = sensor.get_depth_scale()
                    print(f"Derinlik Ölçeği: {depth_scale} metre/birim")
                except Exception as e:
                    print(f"Derinlik ölçeği alınamadı: {e}")
                
        except Exception as e:
            print(f"Bu sensör için bazı veriler okunamıyor: {e}")

def main():
    # RealSense pipeline'ını başlatıyoruz.
    pipeline = rs.pipeline()
    config = rs.config()
    time.sleep(10)
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

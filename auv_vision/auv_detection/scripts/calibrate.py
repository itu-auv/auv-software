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
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams explicitly
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start pipeline FIRST
        profile = pipeline.start(config)
        device = profile.get_device()
        
        # Warm-up AFTER starting
        print("Warming up sensors...")
        time.sleep(10)  # Let hardware stabilize
        
        # Run calibration
        run_self_calibration(device)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()  # Cleanup even on failure
if __name__ == "__main__":
    main()

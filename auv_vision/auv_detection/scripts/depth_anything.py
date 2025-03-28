#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

def show_depth_map_opencv(image_path):
    # 1. Görüntüyü yükle ve kontrol et
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"Hata: {image_path} dosyası bulunamadı veya okunamıyor!")
        return
    
    # 2. Modeli yükle
    model_name = "LiheYoung/depth-anything-small-hf"
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float32)

    # 3. Görüntüyü işleme için hazırla
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    # 4. Derinlik tahmini yap
    inputs = image_processor(images=image_pil, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    
    # 5. Depth map'i orijinal görüntü boyutuna yeniden boyutlandır
    depth_resized = cv2.resize(depth, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    # 6. Normalizasyon ve renklendirme
    depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 7. Boyut kontrolü ve kesme işlemi
    min_height = min(image_cv.shape[0], depth_colored.shape[0])
    min_width = min(image_cv.shape[1], depth_colored.shape[1])
    
    image_cv = image_cv[:min_height, :min_width]
    depth_colored = depth_colored[:min_height, :min_width]
    
    # 8. Görüntüleri yan yana birleştir
    combined = np.hstack((image_cv, depth_colored))
    
    # 9. Görselleştirme
    cv2.imshow("Original (Left) vs Depth Map (Right)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 10. Çıktıyı kaydet
    cv2.imwrite("depth_output.jpg", depth_colored)
    print("Depth map kaydedildi: depth_output.jpg")

if __name__ == "__main__":
    input_image = "input.jpg"  # Aynı dizindeki dosya
    show_depth_map_opencv(input_image)
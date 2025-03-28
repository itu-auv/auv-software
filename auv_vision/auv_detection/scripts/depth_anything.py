#!/usr/bin/env python3
import numpy as np
from PIL import Image
import open3d as o3d
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

# 1. Hafif Model ve İşlemciyi Yükle (CPU Optimize)
model_name = "LiheYoung/depth-anything-small-hf"
image_processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=torch.float32)

# 2. Derinlik Tahmini Fonksiyonu (Low-Memory)
def predict_depth(image_path):
    image = Image.open(image_path)
    inputs = image_processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth
    
    # Min-Max Normalleştirme
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
    depth_uint8 = depth.astype(np.uint8)
    print(depth_uint8)


# 3. Point Cloud Dönüşümü (Basitleştirilmiş)
def create_point_cloud(depth, image, downsample=4):
    h, w = depth.shape
    depth = depth[::downsample, ::downsample]
    rgb = np.array(image.resize((w, h))[::downsample, ::downsample])
    
    # Basit Projeksiyon (FOV 60 varsayılan)
    focal = w / (2 * np.tan(60 * np.pi / 360))
    u, v = np.meshgrid(np.arange(0, w, downsample), np.arange(0, h, downsample))
    z = depth / 255.0  # 0-1 aralığına normalize
    x = (u - w/2) * z / focal
    y = (v - h/2) * z / focal
    
    points = np.stack((x, -y, -z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# 4. Çalıştırma Örneği
depth = predict_depth("input.jpg")


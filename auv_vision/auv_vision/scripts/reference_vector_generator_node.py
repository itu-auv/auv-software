#!/home/frk/miniconda3/envs/auto_label/bin/python
# -*- coding: utf-8 -*-

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from torch import nn, Tensor
from PIL import Image

import rospy
import os
import cv2
import numpy as np
import math
import threading
from sensor_msgs.msg import Image as ROSImage
from std_srvs.srv import Trigger, TriggerResponse
import message_filters
import rospkg

class ResizeToMultiple(nn.Module):
    def __init__(self, short_side: int, multiple: int):
        super().__init__()
        self.short_side = short_side
        self.multiple = multiple

    def _round_up(self, side: float) -> int:
        return math.ceil(side / self.multiple) * self.multiple

    def forward(self, img):
        old_width, old_height = TVTF.get_image_size(img)
        if old_width > old_height:
            new_height = self._round_up(self.short_side)
            new_width = self._round_up(old_width * new_height / old_height)
        else:
            new_width = self._round_up(self.short_side)
            new_height = self._round_up(old_height * new_width / old_width)
        return TVTF.resize(img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC)

class ReferenceVectorGeneratorNode:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_mask = None

        # Parameters
        self.model_path = rospy.get_param(
            "~model_path", "/home/frk/AUTO_LABEL/auv_detection/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        )
        
        # Resolve vectors directory
        r = rospkg.RosPack()
        pkg_path = r.get_path('auv_vision')
        self.vectors_dir = os.path.join(pkg_path, 'vectors')
        if not os.path.exists(self.vectors_dir):
            os.makedirs(self.vectors_dir)

        # Load DINOv3 model
        rospy.loginfo("[ReferenceVectorGeneratorNode] Loading DINOv3 model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=False)
        self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.model.to(self.device)
        self.model.eval()
        self.patch_size = self.model.patch_size

        SHORT_SIDE = 480
        self.transform = TVT.Compose(
            [
                ResizeToMultiple(short_side=SHORT_SIDE, multiple=self.patch_size),
                TVT.ToTensor(),
                TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Subscribers
        self.image_sub = message_filters.Subscriber('/taluy/vision/draw_mask/image', ROSImage)
        self.mask_sub = message_filters.Subscriber('/taluy/vision/draw_mask/mask', ROSImage)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.mask_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.sync_callback)

        # Service Server
        self.srv = rospy.Service('save_reference_vector', Trigger, self.handle_save_reference_vector)

        rospy.loginfo("[ReferenceVectorGeneratorNode] Subscribed to /taluy/vision/draw_mask/image and /taluy/vision/draw_mask/mask")
        rospy.loginfo(f"[ReferenceVectorGeneratorNode] Vectors directory: {self.vectors_dir}")
        rospy.loginfo("[ReferenceVectorGeneratorNode] Service 'save_reference_vector' is ready.")

    def sync_callback(self, img_msg, mask_msg):
        with self.lock:
            self.latest_image = img_msg
            self.latest_mask = mask_msg

    def ros_image_to_numpy(self, img_msg, desired_encoding="bgr8"):
        if img_msg.encoding == "bgr8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            if desired_encoding == "rgb8":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.copy()
        elif img_msg.encoding == "rgb8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
            if desired_encoding == "bgr8":
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img.copy()
        elif img_msg.encoding == "mono8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
            return img.copy()
        else:
            channels = len(img_msg.data) // (img_msg.height * img_msg.width)
            if channels == 3:
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, 3)
                if desired_encoding == "rgb8" and img_msg.encoding.lower().startswith("bgr"):
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif desired_encoding == "bgr8" and img_msg.encoding.lower().startswith("rgb"):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img.copy()
            elif channels == 1:
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width)
                return img.copy()
            else:
                raise ValueError(f"Unsupported ROS image encoding: {img_msg.encoding}")

    def handle_save_reference_vector(self, req):
        with self.lock:
            if self.latest_image is None or self.latest_mask is None:
                return TriggerResponse(success=False, message="No mask and image pair has been received yet.")
            img_msg = self.latest_image
            mask_msg = self.latest_mask

        try:
            class_name = rospy.get_param("~class_name", "object")
            
            # Convert ROS Images
            frame = self.ros_image_to_numpy(img_msg, desired_encoding="bgr8")
            mask = self.ros_image_to_numpy(mask_msg, desired_encoding="mono8")

            if np.sum(mask) == 0:
                return TriggerResponse(success=False, message="The received mask is empty! Paint a region first.")

            # Preprocess frame
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            transformed_frame = self.transform(frame_pil).to(self.device)
            _, frame_height, frame_width = transformed_frame.shape
            feats_height = frame_height // self.patch_size
            feats_width = frame_width // self.patch_size

            # Extract dense features
            with torch.inference_mode():
                feats = self.model.get_intermediate_layers(transformed_frame.unsqueeze(0), n=1, reshape=True)[0]
                feats = feats.movedim(-3, -1)
                feats = F.normalize(feats, dim=-1, p=2)
                first_frame_features = feats.squeeze(0)

            # Interpolate mask
            mask_tensor = torch.from_numpy(mask).to(self.device, dtype=torch.float32)
            mask_tensor[mask_tensor == 255] = 1
            interpolated_mask = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(feats_height, feats_width),
                mode="nearest"
            ).squeeze()
            first_frame_mask = F.one_hot(interpolated_mask.long(), num_classes=2).float()

            # Save the reference vector
            output_file = os.path.join(self.vectors_dir, f"{class_name}.pth")
            save_data = {
                "first_frame_features": first_frame_features.cpu(),
                "first_frame_mask": first_frame_mask.cpu(),
                "class_name": class_name
            }
            torch.save(save_data, output_file)

            msg = f"Successfully generated and saved reference vector for class '{class_name}' to {output_file}"
            rospy.loginfo(msg)
            return TriggerResponse(success=True, message=msg)

        except Exception as e:
            err_msg = f"Error generating reference vector: {str(e)}"
            rospy.logerr(err_msg)
            return TriggerResponse(success=False, message=err_msg)

def main():
    rospy.init_node("reference_vector_generator_node")
    node = ReferenceVectorGeneratorNode()
    rospy.spin()

if __name__ == "__main__":
    main()

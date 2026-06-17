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
from copy import deepcopy
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import Pose2D
from ultralytics_ros.msg import YoloResult
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)
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
        return TVTF.resize(
            img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC
        )


class DinoTrackerNode:
    def __init__(self):

        # Parameters
        self.model_path = rospy.get_param(
            "~model_path",
            "/home/frk/AUTO_LABEL/auv_detection/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        )
        self.yolo_result_topic = rospy.get_param(
            "~yolo_result_topic", "/yolo_result_seg"
        )
        self.default_class_id = int(rospy.get_param("~default_class_id", 0))
        self.class_id_map = {
            str(name): int(class_id)
            for name, class_id in rospy.get_param("~class_id_map", {}).items()
        }

        # Resolve vectors directory
        r = rospkg.RosPack()
        pkg_path = r.get_path("auv_vision")
        self.vectors_dir = os.path.join(pkg_path, "vectors")
        if not os.path.exists(self.vectors_dir):
            os.makedirs(self.vectors_dir)

        # Polling/scanning state
        self.last_dir_scan_time = 0.0
        self.scan_interval = 3.0  # seconds
        self.loaded_vectors = {}
        self.known_files_mtimes = {}

        # Colors setup
        self.colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (192, 192, 192),
            (128, 128, 128),
            (128, 0, 0),
            (128, 128, 0),
            (0, 128, 0),
            (128, 0, 128),
            (0, 128, 128),
            (0, 0, 128),
        ]
        self.class_color_map = {}
        self.color_index = 0

        # Load DINOv3 model
        rospy.loginfo("[DinoTrackerNode] Loading DINOv3 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load(
            "facebookresearch/dinov3", "dinov3_vits16", pretrained=False
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
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

        # First scan of vectors
        self.update_reference_vectors()

        # Publisher
        self.pub_annotated = rospy.Publisher(
            "/taluy/vision/dino_tracker/annotated_image", ROSImage, queue_size=1
        )
        self.pub_yolo_seg = rospy.Publisher(
            self.yolo_result_topic, YoloResult, queue_size=1
        )

        # Subscriber
        self.image_sub = rospy.Subscriber(
            "/taluy/cameras/cam_bottom/image_rect_color",
            ROSImage,
            self.image_callback,
            queue_size=1,
        )

        rospy.loginfo(
            "[DinoTrackerNode] Subscribed to bottom camera: /taluy/cameras/cam_bottom/image_rect_color"
        )
        rospy.loginfo(
            "[DinoTrackerNode] Publishing annotated tracking to /taluy/vision/dino_tracker/annotated_image"
        )
        rospy.loginfo(
            f"[DinoTrackerNode] Publishing DINO masks as YoloResult to {self.yolo_result_topic}"
        )
        rospy.loginfo("[DinoTrackerNode] Initialization complete.")

    def get_class_color(self, class_name):
        if class_name not in self.class_color_map:
            color = self.colors[self.color_index % len(self.colors)]
            self.class_color_map[class_name] = color
            self.color_index += 1
        return self.class_color_map[class_name]

    def update_reference_vectors(self):
        now = rospy.get_time()
        if now - self.last_dir_scan_time < self.scan_interval:
            return
        self.last_dir_scan_time = now

        try:
            files = [f for f in os.listdir(self.vectors_dir) if f.endswith(".pth")]
            current_files_mtimes = {}
            for f in files:
                path = os.path.join(self.vectors_dir, f)
                current_files_mtimes[path] = os.path.getmtime(path)

            # Reload only if files have changed
            if current_files_mtimes != self.known_files_mtimes:
                rospy.loginfo(
                    "[DinoTrackerNode] Reference vectors folder changed, reloading..."
                )
                new_loaded_vectors = {}
                for path, mtime in current_files_mtimes.items():
                    try:
                        data = torch.load(path, map_location="cpu")
                        class_name = data.get(
                            "class_name", os.path.splitext(os.path.basename(path))[0]
                        )

                        first_feats = data["first_frame_features"].to(self.device)
                        first_mask_tensor = data["first_frame_mask"].to(self.device)

                        new_loaded_vectors[class_name] = {
                            "first_frame_features": first_feats,
                            "first_frame_mask": first_mask_tensor,
                            "color": self.get_class_color(class_name),
                        }
                        rospy.loginfo(
                            f"[DinoTrackerNode] Loaded reference vector for class '{class_name}'"
                        )
                    except Exception as e:
                        rospy.logerr(
                            f"[DinoTrackerNode] Failed to load vector {path}: {e}"
                        )

                self.loaded_vectors = new_loaded_vectors
                self.known_files_mtimes = current_files_mtimes
        except Exception as e:
            rospy.logerr_throttle(
                10.0, f"[DinoTrackerNode] Error scanning vectors folder: {e}"
            )

    def extract_dense_features(self, image_tensor: Tensor) -> Tensor:
        with torch.inference_mode():
            feats = self.model.get_intermediate_layers(
                image_tensor.unsqueeze(0), n=1, reshape=True
            )[0]
            feats = feats.movedim(-3, -1)
            feats = F.normalize(feats, dim=-1, p=2)
            return feats.squeeze(0)

    def propagate_mask(
        self, current_feats: Tensor, first_feats: Tensor, first_mask_tensor: Tensor
    ) -> Tensor:
        TOPK = 5
        TEMPERATURE = 0.2
        h, w, M = first_mask_tensor.shape
        h_curr, w_curr, _ = current_feats.shape
        context_features = first_feats.unsqueeze(0)
        context_probs = first_mask_tensor.unsqueeze(0)
        dot = torch.einsum("ijd, tuvd -> ijtuv", current_feats, context_features)
        dot = dot.flatten(2, -1).flatten(0, 1)
        k_th_largest = torch.topk(dot, dim=1, k=TOPK).values
        dot = torch.where(dot >= k_th_largest[:, -1:], dot, -torch.inf)
        weights = F.softmax(dot / TEMPERATURE, dim=1)
        current_probs = torch.mm(weights, context_probs.flatten(0, 2))
        current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)
        return current_probs.unflatten(0, (h_curr, w_curr))

    def ros_image_to_numpy(self, img_msg, desired_encoding="bgr8"):
        if img_msg.encoding == "bgr8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
            if desired_encoding == "rgb8":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.copy()
        elif img_msg.encoding == "rgb8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
            if desired_encoding == "bgr8":
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img.copy()
        elif img_msg.encoding == "mono8":
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width
            )
            return img.copy()
        else:
            channels = len(img_msg.data) // (img_msg.height * img_msg.width)
            if channels == 3:
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                    img_msg.height, img_msg.width, 3
                )
                if desired_encoding == "rgb8" and img_msg.encoding.lower().startswith(
                    "bgr"
                ):
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif desired_encoding == "bgr8" and img_msg.encoding.lower().startswith(
                    "rgb"
                ):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img.copy()
            elif channels == 1:
                img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                    img_msg.height, img_msg.width
                )
                return img.copy()
            else:
                raise ValueError(f"Unsupported ROS image encoding: {img_msg.encoding}")

    def numpy_to_ros_image(self, cv_image, encoding="bgr8"):
        img_msg = ROSImage()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        img_msg.is_bigendian = 0
        if len(cv_image.shape) == 3:
            img_msg.step = cv_image.shape[1] * cv_image.shape[2]
        else:
            img_msg.step = cv_image.shape[1]
        img_msg.data = cv_image.tobytes()
        return img_msg

    def make_detection(self, class_id, x, y, w, h):
        det = Detection2D()
        hyp = ObjectHypothesisWithPose()
        hyp.id = class_id
        hyp.score = 1.0
        det.results.append(hyp)
        det.bbox = BoundingBox2D()
        det.bbox.center = Pose2D(x=float(x + w / 2.0), y=float(y + h / 2.0), theta=0.0)
        det.bbox.size_x = float(w)
        det.bbox.size_y = float(h)
        return det

    def make_yolo_result(self, header, detections=None, masks=None):
        result_msg = YoloResult()
        result_msg.header = header
        result_msg.detections = Detection2DArray()
        result_msg.detections.header = header
        result_msg.detections.detections = detections or []
        result_msg.masks = masks or []
        return result_msg

    def image_callback(self, msg):
        self.update_reference_vectors()

        if not self.loaded_vectors:
            rospy.loginfo_throttle(
                10.0,
                "[DinoTrackerNode] No reference vectors loaded. Publish original stream.",
            )
            self.pub_annotated.publish(msg)
            self.pub_yolo_seg.publish(self.make_yolo_result(msg.header))
            return

        try:
            # Convert ROS Image to OpenCV
            frame_bgr = self.ros_image_to_numpy(msg, desired_encoding="bgr8")
            vis_dino = frame_bgr.copy()
            detections = []
            masks = []

            # Preprocess image
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            transformed_frame = self.transform(frame_pil).to(self.device)
            current_feats = self.extract_dense_features(transformed_frame)

            for class_name, data in self.loaded_vectors.items():
                first_frame_features = data["first_frame_features"]
                first_frame_mask = data["first_frame_mask"]
                class_color_bgr = data["color"]

                # Propagate mask
                propagated_mask_probs = self.propagate_mask(
                    current_feats, first_frame_features, first_frame_mask
                )
                propagated_mask_upsampled = cv2.resize(
                    torch.argmax(propagated_mask_probs, dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                    (frame_bgr.shape[1], frame_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Overlay predicted mask
                mask_vis_overlay = np.zeros_like(frame_bgr)
                mask_vis_overlay[propagated_mask_upsampled == 1] = class_color_bgr
                vis_dino = cv2.addWeighted(vis_dino, 1.0, mask_vis_overlay, 0.4, 0)

                # Find labels & bounding boxes
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                    propagated_mask_upsampled, connectivity=8
                )
                if num_labels > 1:
                    largest_component_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    x = stats[largest_component_label, cv2.CC_STAT_LEFT]
                    y = stats[largest_component_label, cv2.CC_STAT_TOP]
                    w = stats[largest_component_label, cv2.CC_STAT_WIDTH]
                    h = stats[largest_component_label, cv2.CC_STAT_HEIGHT]

                    # Draw Bounding Box & Class Name
                    cv2.rectangle(vis_dino, (x, y), (x + w, y + h), class_color_bgr, 2)
                    cv2.putText(
                        vis_dino,
                        class_name,
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        class_color_bgr,
                        2,
                    )

                    class_id = self.class_id_map.get(class_name, self.default_class_id)
                    mask_img = (propagated_mask_upsampled == 1).astype(np.uint8) * 255
                    mask_msg = self.numpy_to_ros_image(mask_img, encoding="mono8")
                    mask_msg.header = deepcopy(msg.header)
                    mask_msg.header.frame_id = str(class_id)

                    detections.append(self.make_detection(class_id, x, y, w, h))
                    masks.append(mask_msg)

            # Publish annotated tracking results
            annotated_msg = self.numpy_to_ros_image(vis_dino, encoding="bgr8")
            annotated_msg.header = msg.header
            self.pub_annotated.publish(annotated_msg)

            self.pub_yolo_seg.publish(
                self.make_yolo_result(msg.header, detections, masks)
            )

        except Exception as e:
            rospy.logerr_throttle(
                5.0, f"[DinoTrackerNode] Error during tracking callback: {e}"
            )


def main():
    rospy.init_node("dino_tracker_node")
    node = DinoTrackerNode()
    rospy.spin()


if __name__ == "__main__":
    main()

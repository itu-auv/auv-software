#!/usr/bin/env python3

import cv2
import rospy
import torch
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
import torch.nn as nn
import torch.nn.functional as Fnn


def downsample_to_target_avgpool(F: torch.Tensor, target: int = 32) -> torch.Tensor:
    _, _, H, W = F.shape

    kh = H // target
    kw = W // target

    if kh == 0 or kw == 0:
        return F

    if (H % target) != 0 or (W % target) != 0:
        H2 = kh * target
        W2 = kw * target
        H2 = max(H2, target)
        W2 = max(W2, target)

        F = Fnn.interpolate(F, size=(H2, W2), mode="bilinear", align_corners=False)

        H, W = H2, W2
        kh = H // target
        kw = W // target

    return Fnn.avg_pool2d(F, kernel_size=(kh, kw), stride=(kh, kw))


class FSTS(nn.Module):
    def __init__(self, block1, channels):
        super(FSTS, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias


class FGDPAUIENetS(nn.Module):
    def __init__(self, channels, fft_size=32):
        super(FGDPAUIENetS, self).__init__()
        self.channels = channels
        self.fft_size = fft_size

        self.head = FSTS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1),
            ),
            channels,
        )

        self.body = FSTS(nn.Conv2d(channels, channels, 3, 1, 1), channels)

        # fgdpa convs (slim target)
        self.fgdpa_fca = nn.Conv2d(2 * channels, channels, 1, 1)
        self.fgdpa_fgsa = nn.Conv2d(2, channels, 1, 1)

        # gate params (loaded from slim)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.lam = nn.Parameter(torch.tensor(0.5))

        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)

    def _fgdpa_attention(self, F: torch.Tensor) -> torch.Tensor:
        B, C, H, W = F.shape

        F_ds = downsample_to_target_avgpool(F, target=self.fft_size)

        freq = torch.fft.fft2(F_ds)
        freq_mag = torch.log1p(torch.abs(freq))

        M = freq_mag

        max_map, _ = torch.max(F, dim=1, keepdim=True)
        avg_map = torch.mean(F, dim=1, keepdim=True)

        fgdpa_spatial_in = torch.cat([max_map, avg_map], dim=1)

        A_s = torch.sigmoid(self.fgdpa_fgsa(fgdpa_spatial_in))

        gap_F = torch.mean(F, dim=(2, 3), keepdim=True)

        gap_M = torch.mean(M, dim=(2, 3), keepdim=True)
        gap_M = gap_M / (gap_M.mean(dim=1, keepdim=True) + 1e-6)

        fgdpa_channel_in = torch.cat([gap_F, gap_M], dim=1)

        A_c = torch.sigmoid(self.fgdpa_fca(fgdpa_channel_in))

        Ag, Al = A_c, A_s
        lam = torch.clamp(self.lam, 0.0, 1.0)

        A_lin = self.alpha * Ag + self.beta * Al
        A_int = Ag * Al
        A = (1.0 - lam) * A_lin + lam * A_int

        return A

    def forward(self, x):
        x0 = self.head(x)
        F = self.body(x0)

        A = self._fgdpa_attention(F)
        F_hat = A * F

        return self.tail(F_hat)


class FGDPANode:
    def __init__(self):
        rospy.init_node("fgdpa_enhancement_node", anonymous=False)

        # Parameters
        self.model_path = rospy.get_param("~model_path", None)
        self.channels = rospy.get_param("~channels", 12)
        self.input_topic = rospy.get_param("~input_topic", "/camera/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/fgdpa/enhanced_image")
        self.device_str = rospy.get_param(
            "~device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Device setup
        self.device = torch.device(self.device_str)
        rospy.loginfo(f"[FGDPA] Using device: {self.device}")

        # Model setup
        self.model = None
        self.is_loaded = False
        self.bridge = CvBridge()
        self.output_compressed_topic = self.output_topic + "/compressed"

        if self.model_path and self._load_model():
            rospy.loginfo(f"[FGDPA] Model loaded from: {self.model_path}")
        else:
            rospy.logwarn("[FGDPA] No valid model path provided or loading failed")
            rospy.logwarn("[FGDPA] Set _model_path parameter in launch file")

        self.sub_image = rospy.Subscriber(
            self.input_topic, Image, self.image_callback, queue_size=1
        )

        self.pub_image = rospy.Publisher(self.output_topic, Image, queue_size=1)

        self.pub_compressed = rospy.Publisher(
            self.output_compressed_topic, CompressedImage, queue_size=1
        )

        rospy.loginfo(f"[FGDPA] Node initialized")
        rospy.loginfo(f"[FGDPA] Subscribing to: {self.input_topic}")
        rospy.loginfo(f"[FGDPA] Publishing to: {self.output_topic}")

    def _load_model(self):
        if not self.model_path:
            return False

        try:
            self.model = FGDPAUIENetS(channels=self.channels).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            rospy.logerr(f"[FGDPA] Failed to load model: {e}")
            return False

    def image_callback(self, msg):
        if not self.is_loaded or self.model is None:
            rospy.logwarn_throttle(5, "[FGDPA] Model not loaded, skipping frame")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            img_np = cv_image.astype(np.float32) / 255.0
            img_tensor = (
                torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                output = self.model(img_tensor)

            output_clamped = output.clamp(0, 1)[0]
            output_np = (output_clamped.permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )

            output = self.bridge.cv2_to_imgmsg(output_np, encoding="rgb8")
            output.header = msg.header
            self.pub_image.publish(output)

            compressed_msg = CompressedImage()
            compressed_msg.header = msg.header
            compressed_msg.format = "jpeg"

            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

            compressed_msg.data = np.array(
                cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
            ).tobytes()
            self.pub_compressed.publish(compressed_msg)

            rospy.logdebug("[FGDPA] Frame processed")

        except CvBridgeError as e:
            rospy.logerr(f"[FGDPA] CV Bridge error: {e}")
        except Exception as e:
            rospy.logerr(f"[FGDPA] Processing error: {e}")


if __name__ == "__main__":
    node = FGDPANode()
    rospy.spin()

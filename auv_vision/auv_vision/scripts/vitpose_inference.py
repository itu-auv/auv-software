#!/usr/bin/env python3

"""
Self-contained ViTPose-B inference for the valve dataset.
Dependencies: torch, numpy, cv2  — nothing else.

Usage:
    from vitpose_inference import ValvePose
    model = ValvePose('path/to/best.pth')
    kps, scores = model.predict(img_rgb, bbox_xywh)
    # kps:    np.ndarray (8, 2)  — pixel coords (x, y)
    # scores: np.ndarray (8, 1)  — confidence
"""

import math
import os
import warnings
import collections.abc
from itertools import repeat
from functools import partial
import glob

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# ══════════════════════════════════════════════════════════════════════════════
# ViT Backbone
# ══════════════════════════════════════════════════════════════════════════════


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.num_patches = self.patch_shape[0] * self.patch_shape[1] * (ratio**2)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
        )

    def forward(self, x):
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        return x.view(B, C, Hp * Wp).transpose(1, 2), (Hp, Wp)


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_checkpoint=False,
        ratio=1,
        last_norm=True,
        **kwargs,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            ratio=ratio,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).view(B, -1, Hp, Wp).contiguous()


# ══════════════════════════════════════════════════════════════════════════════
# Keypoint Head
# ══════════════════════════════════════════════════════════════════════════════


class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        layers = []
        for i in range(num_deconv_layers):
            k = num_deconv_kernels[i]
            padding = 1 if k == 4 else (1 if k == 3 else 0)
            out_pad = 0 if k == 4 else (1 if k == 3 else 0)
            planes = num_deconv_filters[i]
            layers += [
                nn.ConvTranspose2d(
                    self.in_channels,
                    planes,
                    kernel_size=k,
                    stride=2,
                    padding=padding,
                    output_padding=out_pad,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
            self.in_channels = planes
        self.deconv_layers = nn.Sequential(*layers)

        final_kernel = extra.get("final_conv_kernel", 1) if extra else 1
        padding = 1 if final_kernel == 3 else 0
        self.final_layer = nn.Conv2d(
            self.in_channels,
            out_channels,
            kernel_size=final_kernel,
            stride=1,
            padding=padding,
        )

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


# ══════════════════════════════════════════════════════════════════════════════
# Full model
# ══════════════════════════════════════════════════════════════════════════════


class ViTPose(nn.Module):
    def __init__(self, backbone_cfg, head_cfg):
        super().__init__()
        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)

    def forward(self, x):
        return self.keypoint_head(self.backbone(x))


# ── default configs ───────────────────────────────────────────────────────────
BACKBONE_CFG_B = dict(
    img_size=(320, 256),
    patch_size=16,
    embed_dim=768,
    depth=12,
    num_heads=12,
    ratio=1,
    use_checkpoint=False,
    mlp_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.3,
)
BACKBONE_CFG_L = dict(
    img_size=(320, 256),
    patch_size=16,
    embed_dim=1024,
    depth=24,
    num_heads=16,
    ratio=1,
    use_checkpoint=False,
    mlp_ratio=4,
    qkv_bias=True,
    drop_path_rate=0.3,
)
# Keep BACKBONE_CFG as alias for backward compatibility
BACKBONE_CFG = BACKBONE_CFG_B
HEAD_CFG = dict(
    in_channels=768,
    out_channels=8,
    num_deconv_layers=2,
    num_deconv_filters=(256, 256),
    num_deconv_kernels=(4, 4),
    extra=dict(final_conv_kernel=1),
)


# ══════════════════════════════════════════════════════════════════════════════
# Pre / post processing
# ══════════════════════════════════════════════════════════════════════════════

IMG_W, IMG_H = 256, 320
PIXEL_STD = 200
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_3rd_point(a, b):
    d = a - b
    return b + np.array([-d[1], d[0]], dtype=np.float32)


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return [pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs]


def get_affine_transform(center, scale, pixel_std, rot, output_size):
    scale_tmp = scale * pixel_std
    src_w = scale_tmp[0]
    dst_w, dst_h = output_size
    rot_rad = np.pi * rot / 180
    src_dir = np.array(_rotate_point([0.0, src_w * -0.5], rot_rad), dtype=np.float32)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + src_dir
    src[2] = _get_3rd_point(src[0], src[1])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = dst[0] + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _get_max_preds(heatmaps):
    N, K, _, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    idx = np.argmax(flat, 2).reshape(N, K, 1)
    maxvals = np.amax(flat, 2).reshape(N, K, 1)
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1]
            - heatmap[py - 1][px + 1]
            - heatmap[py + 1][px - 1]
            + heatmap[py - 1][px - 1]
        )
        dyy = 0.25 * (heatmap[py + 2][px] - 2 * heatmap[py][px] + heatmap[py - 2][px])
        derivative = np.array([dx, dy])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            coord += -np.linalg.inv(hessian) @ derivative
    return coord


def _gaussian_blur(heatmaps, kernel=11):
    border = (kernel - 1) // 2
    N, K, H, W = heatmaps.shape
    for i in range(N):
        for j in range(K):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border]
            heatmaps[i, j] *= origin_max / (np.max(heatmaps[i, j]) + 1e-9)
    return heatmaps


def transform_preds(coords, center, scale, output_size):
    scale_x = scale[0] / (output_size[0] - 1.0)
    scale_y = scale[1] / (output_size[1] - 1.0)
    target = np.ones_like(coords)
    target[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target


def keypoints_from_heatmaps(heatmaps, center, scale, kernel=11):
    """Unbiased post-processing (UDP + DARK). Returns (preds, maxvals)."""
    heatmaps = heatmaps.copy()
    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)

    # DARK refinement
    heatmaps_log = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
    for n in range(N):
        for k in range(K):
            preds[n][k] = _taylor(heatmaps_log[n][k], preds[n][k])

    # map back to image coords (UDP)
    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H])

    return preds, maxvals


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

# Adjacent pairs connecting the 8 bolt-hole keypoints in a circular ring.
# Indices 0–7 are arranged at 45° increments around the valve face.
# Index 8 = valve face origin, index 9 = valve stem tip (if present).
SKELETON_8 = [(i, (i + 1) % 8) for i in range(8)]
SKELETON_10 = SKELETON_8 + [(8, 9)]  # origin → stem tip
SKELETON = SKELETON_8  # default for backward compat

# Horizontal-flip keypoint swap map for TTA.
# Bolt holes sit at 45° increments: -135, -90, -45, 0, 45, 90, 135, 180.
# Flipping mirrors θ → -θ:  0↔2, 3↔7, 4↔6; 1 and 5 stay.
FLIP_INDEX = [2, 1, 0, 7, 6, 5, 4, 3]


class ValvePose:
    """
    Load once, call predict() per frame.

    Args:
        ckpt:   path to best.pth, or a runs/train directory (picks latest automatically)
        device: 'cuda' | 'cpu'
    """

    def __init__(self, ckpt: str, device: str = "cuda"):
        import os

        if os.path.isdir(ckpt):
            candidates = sorted(glob.glob(os.path.join(ckpt, "*/best.pth")))
            if not candidates:
                raise FileNotFoundError(f"No best.pth found under {ckpt}")
            ckpt = candidates[-1]

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        state = torch.load(ckpt, map_location="cpu", weights_only=True)

        # Auto-detect architecture from checkpoint
        embed_dim = state["backbone.patch_embed.proj.bias"].shape[0]
        num_kps = state["keypoint_head.final_layer.bias"].shape[0]
        if embed_dim == 1024:
            backbone_cfg = dict(BACKBONE_CFG_L)
            variant = "L"
        else:
            backbone_cfg = dict(BACKBONE_CFG_B)
            variant = "B"
        head_cfg = dict(HEAD_CFG, in_channels=embed_dim, out_channels=num_kps)

        self.num_kps = num_kps
        self.model = ViTPose(backbone_cfg, head_cfg)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(
            f"ValvePose loaded: {ckpt}  (ViT-{variant}, {num_kps} kps, {self.device})"
        )

    @torch.no_grad()
    def predict(self, img_rgb: np.ndarray, bbox_xywh, flip: bool = True):
        """
        Args:
            img_rgb:   H×W×3 uint8 RGB image
            bbox_xywh: (x, y, w, h) bounding box in pixel coords
            flip:      enable horizontal-flip TTA (averages heatmaps)

        Returns:
            kps    np.ndarray (8, 2)  pixel coords (x, y)
            scores np.ndarray (8, 1)  confidence [0, 1]
        """
        x, y, w, h = bbox_xywh
        cx, cy = x + w * 0.5, y + h * 0.5
        aspect = IMG_W / IMG_H
        if w > aspect * h:
            h = w / aspect
        elif w < aspect * h:
            w = h * aspect
        scale = np.array([w / PIXEL_STD, h / PIXEL_STD], dtype=np.float32) * 1.25
        center = np.array([cx, cy], dtype=np.float32)

        trans = get_affine_transform(center, scale, PIXEL_STD, 0, (IMG_W, IMG_H))
        crop = cv2.warpAffine(img_rgb, trans, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR)
        inp = (
            torch.from_numpy(
                ((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
            )
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        heatmaps = self.model(inp).cpu().numpy()

        # Flip TTA only when FLIP_INDEX covers all channels (8-kp models)
        if flip and self.num_kps == 8:
            crop_flip = crop[:, ::-1, :].copy()
            inp_flip = (
                torch.from_numpy(
                    ((crop_flip.astype(np.float32) / 255.0 - MEAN) / STD).transpose(
                        2, 0, 1
                    )
                )
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            hm_flip = self.model(inp_flip).cpu().numpy()
            # Mirror heatmaps back spatially and swap keypoint channels
            hm_flip = hm_flip[:, :, :, ::-1].copy()
            hm_flip = hm_flip[:, FLIP_INDEX, :, :]
            heatmaps = (heatmaps + hm_flip) * 0.5

        preds, scores = keypoints_from_heatmaps(
            heatmaps, center[None], (scale * PIXEL_STD)[None]
        )
        return preds[0], scores[0]


# ---------------------------------------------------------------------------
# TensorRT runtime path (.engine files, AUV/Orin)
# ---------------------------------------------------------------------------
#
# Same predict() signature as ValvePose so consumers don't care which backend
# loaded.  TRT/pycuda imports are deferred until the class is actually
# instantiated — pure-PyTorch deployments (laptop, CI) never trigger them and
# therefore don't need TensorRT installed.

# pycuda's primary CUDA context can only be made once per process; subsequent
# ValvePoseTRT instances reuse it.  Stored as module state so it survives
# multiple imports / re-instantiations.
_TRT_PRIMARY_CTX = None


def _ensure_pycuda_context():
    """Lazy one-time pycuda init.  Returns the (cuda, trt) module handles."""
    global _TRT_PRIMARY_CTX
    import tensorrt as trt
    import pycuda.driver as cuda

    if _TRT_PRIMARY_CTX is None:
        # Manual init — pycuda.autoinit pulls in pycuda.tools / compyte.dtypes
        # which is broken on Python 3.8 with newer pycuda wheels.
        cuda.init()
        _TRT_PRIMARY_CTX = cuda.Device(0).make_context()
    return cuda, trt


class ValvePoseTRT:
    """TensorRT-backed ValvePose (drop-in for ValvePose).

    Args:
        engine_path: path to the built .engine file (device-specific).
        num_kps:     number of keypoint channels in the engine's output.
                     Must match the checkpoint used to export the ONNX.
    """

    def __init__(self, engine_path: str, num_kps: int = 10):
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")

        cuda, trt = _ensure_pycuda_context()
        self._cuda = cuda
        self._trt = trt

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.num_kps = int(num_kps)
        self.in_shape = (1, 3, IMG_H, IMG_W)  # 1×3×320×256
        self.out_shape = (1, self.num_kps, IMG_H // 4, IMG_W // 4)  # 1×N×80×64

        # Pinned host buffers + device buffers + stream.
        self.h_in = cuda.pagelocked_empty(int(np.prod(self.in_shape)), dtype=np.float32)
        self.h_out = cuda.pagelocked_empty(
            int(np.prod(self.out_shape)), dtype=np.float32
        )
        self.d_in = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)
        self.stream = cuda.Stream()

        # Bind IO. TRT 10.x uses named tensors + execute_async_v3;
        # TRT 8.x uses a positional bindings list + execute_async_v2.
        if hasattr(self.engine, "num_io_tensors"):
            names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            in_name = next(
                n
                for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            )
            out_name = next(
                n
                for n in names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            )
            self.context.set_tensor_address(in_name, int(self.d_in))
            self.context.set_tensor_address(out_name, int(self.d_out))
            self._execute = lambda: self.context.execute_async_v3(self.stream.handle)
        else:
            self._bindings = [int(self.d_in), int(self.d_out)]
            self._execute = lambda: self.context.execute_async_v2(
                self._bindings, self.stream.handle
            )

        print(
            f"ValvePoseTRT loaded: {engine_path}  ({self.num_kps} kps, "
            f"in={self.in_shape}, out={self.out_shape})"
        )

    # ------------------------------------------------------------------ public

    def predict(self, img_rgb: np.ndarray, bbox_xywh, flip: bool = True):
        """Drop-in replacement for ValvePose.predict.

        Args:
            img_rgb:   H×W×3 uint8 RGB image
            bbox_xywh: (x, y, w, h) bbox in pixel coords
            flip:      enable horizontal-flip TTA (averages heatmaps).
                       Auto-disabled unless num_kps == 8 (FLIP_INDEX is 8-long).

        Returns:
            preds  np.ndarray (N, 2)  pixel coords
            scores np.ndarray (N, 1)  confidence
        """
        x, y, w, h = bbox_xywh
        cx, cy = x + w * 0.5, y + h * 0.5
        aspect = IMG_W / IMG_H
        if w > aspect * h:
            h = w / aspect
        elif w < aspect * h:
            w = h * aspect
        scale = np.array([w / PIXEL_STD, h / PIXEL_STD], dtype=np.float32) * 1.25
        center = np.array([cx, cy], dtype=np.float32)

        trans = get_affine_transform(center, scale, PIXEL_STD, 0, (IMG_W, IMG_H))
        crop = cv2.warpAffine(img_rgb, trans, (IMG_W, IMG_H), flags=cv2.INTER_LINEAR)
        inp = ((crop.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        inp = np.ascontiguousarray(inp[None])

        heatmaps = self._infer(inp)

        # Flip TTA only when FLIP_INDEX is valid (8-kp layout).
        if flip and self.num_kps == 8:
            crop_flip = crop[:, ::-1, :].copy()
            inp_flip = ((crop_flip.astype(np.float32) / 255.0 - MEAN) / STD).transpose(
                2, 0, 1
            )
            inp_flip = np.ascontiguousarray(inp_flip[None])
            hm_flip = self._infer(inp_flip)
            hm_flip = hm_flip[:, :, :, ::-1].copy()
            hm_flip = hm_flip[:, FLIP_INDEX, :, :]
            heatmaps = (heatmaps + hm_flip) * 0.5

        preds, scores = keypoints_from_heatmaps(
            heatmaps, center[None], (scale * PIXEL_STD)[None]
        )
        return preds[0], scores[0]

    # ------------------------------------------------------------------ internal

    def _infer(self, x_np: np.ndarray) -> np.ndarray:
        cuda = self._cuda
        np.copyto(self.h_in, x_np.ravel())
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self._execute()
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out.reshape(self.out_shape).copy()


# ---------------------------------------------------------------------------
# Backend-agnostic loader
# ---------------------------------------------------------------------------


def load_valve_pose(ckpt: str, device: str = "cuda", num_kps: int = 10):
    """Return a ValvePose-compatible object based on the file extension.

    `.engine`           → ValvePoseTRT  (TensorRT runtime, GPU only)
    `.pth` / `.pt` /    → ValvePose     (PyTorch, CPU or CUDA)
    directory of best.pth

    Same predict() signature either way, so the rest of the pipeline doesn't
    have to know which backend loaded.
    """
    if ckpt.endswith(".engine"):
        return ValvePoseTRT(ckpt, num_kps=num_kps)
    return ValvePose(ckpt, device=device)

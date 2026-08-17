#!/usr/bin/env python3
"""Self-contained ViTPose inference runtimes: joint (pose + seg) and objectness.

Two models, one ViT trunk, one file (house rule; apart from vitpose_utils.py
only because it imports torch). Faithful ports of the valve-vision reference
— parity against it is the acceptance test. Full contract (preprocess,
decode, thresholds): gate_tetra_overview.md.

    model = load_vitpose("gate_joint.pth", device="cuda")
    kps, scores, mask_probs = model.predict(img_rgb, bbox_xywh)
    # kps (K, 2) source px; scores (K, 1); mask_probs (C, H_src, W_src) | None

    detector = load_objectness("tetra_objectness.pth")
    bbox_xywh, score = detector.predict(img_rgb)   # bbox None = no detection

Auto-configured from the checkpoint payload (K, C, ViT size, input size,
mask_threshold). Input resolution is the checkpoint's own — the pos-embed is
trained at that size; a different resolution means training a model at it.
Everything is RGB. Decode default use_udp=False is contractual: the heads are
MSRA-encoded and UDP decode measured ~3x worse (7.38 vs 2.38 px).

Dependencies: torch, numpy, cv2 — nothing else (tensorrt only for .engine
files; see the TensorRT section + utils/vitpose_export.py).
"""

import collections.abc
import json
import math
import os
from functools import partial
from itertools import repeat

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PIXEL_STD = 200.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# embed_dim -> ViT topology (mirrors valve-vision core/model_factory.py).
# ViT-S uses num_heads=12 (a 32-wide head), NOT DeiT's more common 6: the
# published easy_ViTPose weights were trained that way, and 6 partitions qkv
# differently for identical tensor shapes — it loads without error and quietly
# destroys the pretrained attention.
_ARCHITECTURES = {
    384: dict(depth=12, num_heads=12, drop_path_rate=0.1),  # ViT-S
    768: dict(depth=12, num_heads=12, drop_path_rate=0.3),  # ViT-B
    1024: dict(depth=24, num_heads=16, drop_path_rate=0.5),  # ViT-L
    1280: dict(depth=32, num_heads=16, drop_path_rate=0.55),  # ViT-H
}
PATCH_SIZE = 16


# ══════════════════════════════════════════════════════════════════════════════
# ViT backbone (faithful port of easy_vitpose vit_models/backbone/vit.py —
# state-dict compatible with valve-vision joint checkpoints)
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

    low = norm_cdf((a - mean) / std)
    up = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * low - 1, 2 * up - 1)
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
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
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
        img_size=(256, 192),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        ratio=1,
        last_norm=True,
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
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
        # First (cls) pos-embed token is added everywhere, matching ViTPose.
        x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
        for blk in self.blocks:
            x = blk(x)
        x = self.last_norm(x)
        return x.permute(0, 2, 1).view(B, -1, Hp, Wp).contiguous()


# ══════════════════════════════════════════════════════════════════════════════
# Heatmap / mask head
# ══════════════════════════════════════════════════════════════════════════════


class TopdownHeatmapSimpleHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        final_conv_kernel=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        layers = []
        for i in range(num_deconv_layers):
            kernel = num_deconv_kernels[i]
            padding = {4: 1, 3: 1, 2: 0}[kernel]
            output_padding = {4: 0, 3: 1, 2: 0}[kernel]
            planes = num_deconv_filters[i]
            layers += [
                nn.ConvTranspose2d(
                    self.in_channels,
                    planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
            ]
            self.in_channels = planes
        self.deconv_layers = nn.Sequential(*layers)
        padding = 1 if final_conv_kernel == 3 else 0
        self.final_layer = nn.Conv2d(
            self.in_channels,
            out_channels,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=padding,
        )

    def forward(self, x):
        return self.final_layer(self.deconv_layers(x))


def _head_cfg_from_state(state, prefix, in_channels):
    """Rebuild a head's constructor args from its state-dict tensor shapes.

    Deconv structure (layer count, filters, kernels) and the final conv are
    read from the weights themselves, so any head configuration a checkpoint
    was trained with loads without matching config files.
    """
    deconvs = []
    for key in sorted(
        (k for k in state if k.startswith(f"{prefix}.deconv_layers.")),
        key=lambda k: int(k.split(".")[2]),
    ):
        tensor = state[key]
        if key.endswith(".weight") and tensor.dim() == 4:
            # ConvTranspose2d weight: (in, out, kH, kW)
            deconvs.append((int(tensor.shape[1]), int(tensor.shape[2])))
    final_w = state[f"{prefix}.final_layer.weight"]
    return dict(
        in_channels=in_channels,
        out_channels=int(final_w.shape[0]),
        num_deconv_layers=len(deconvs),
        num_deconv_filters=tuple(f for f, _ in deconvs),
        num_deconv_kernels=tuple(k for _, k in deconvs),
        final_conv_kernel=int(final_w.shape[2]),
    )


class JointVitpose(nn.Module):
    """ViT trunk + pose head + optional mask head; attribute names match the
    checkpoint state dict (`backbone.* / pose_head.* / mask_head.*`)."""

    def __init__(self, backbone_cfg, pose_head_cfg, mask_head_cfg=None):
        super().__init__()
        self.backbone = ViT(**backbone_cfg)
        self.pose_head = TopdownHeatmapSimpleHead(**pose_head_cfg)
        self.mask_head = (
            TopdownHeatmapSimpleHead(**mask_head_cfg) if mask_head_cfg else None
        )
        self._img_size = tuple(backbone_cfg["img_size"])  # (H, W)

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.pose_head(features)
        mask_logits = None
        if self.mask_head is not None:
            mask_logits = self.mask_head(features)
            target = (self._img_size[0] // 4, self._img_size[1] // 4)
            if mask_logits.shape[-2:] != target:
                mask_logits = F.interpolate(
                    mask_logits, size=target, mode="bilinear", align_corners=False
                )
        return heatmaps, mask_logits


# ══════════════════════════════════════════════════════════════════════════════
# Preprocess (gate_tetra_overview.md §1.1)
# ══════════════════════════════════════════════════════════════════════════════


def box2cs(bbox_xywh, input_width, input_height):
    """ViTPose aspect-fit crop with the contractually fixed 1.25x pad."""
    x, y, width, height = (float(v) for v in bbox_xywh)
    center = np.asarray([x + width * 0.5, y + height * 0.5], dtype=np.float32)
    aspect = input_width / input_height
    if width > aspect * height:
        height = width / aspect
    elif width < aspect * height:
        width = height * aspect
    scale = np.asarray([width / PIXEL_STD, height / PIXEL_STD], dtype=np.float32)
    return center, scale * 1.25


def _get_3rd_point(a, b):
    d = a - b
    return b + np.array([-d[1], d[0]], dtype=np.float32)


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    return [pt[0] * cs - pt[1] * sn, pt[0] * sn + pt[1] * cs]


def get_affine_transform(center, scale, pixel_std, rot, output_size):
    """Standard ViTPose/MMPose top-down affine; output_size is (W, H)."""
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


# ══════════════════════════════════════════════════════════════════════════════
# Keypoint decode (mmpose-faithful port; gate_tetra_overview.md §1.2)
# ══════════════════════════════════════════════════════════════════════════════


def _get_max_preds(heatmaps):
    N, K, _, W = heatmaps.shape
    flat = heatmaps.reshape((N, K, -1))
    idx = np.argmax(flat, 2).reshape((N, K, 1))
    maxvals = np.amax(flat, 2).reshape((N, K, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals


def _taylor(heatmap, coord):
    """DARK distribution-aware sub-pixel refinement of one peak."""
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
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy**2 != 0:
            offset = -np.linalg.inv(hessian) @ derivative
            coord += np.squeeze(np.array(offset.T), axis=0)
    return coord


def _gaussian_blur(heatmaps, kernel=11):
    """Gaussian modulation preserving each map's peak value (DARK prep)."""
    assert kernel % 2 == 1
    border = (kernel - 1) // 2
    N, K, H, W = heatmaps.shape
    for i in range(N):
        for j in range(K):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """UDP-style DARK refinement (used only when use_udp=True)."""
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert B == 1 or B == N
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)
    pad = np.pad(batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="edge")
    pad = pad.flatten()
    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = pad[index]
    ix1 = pad[index + 1]
    iy1 = pad[index + W + 2]
    ix1y1 = pad[index + W + 3]
    ix1_y1_ = pad[index - W - 3]
    ix1_ = pad[index - 1]
    iy1_ = pad[index - 2 - W]
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1).reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1).reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum("ijmn,ijnk->ijmk", hessian, derivative).squeeze()
    return coords


def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Heatmap coords -> source pixels. use_udp picks /(size-1) vs /size —
    this single line is the 3x accuracy difference (module docstring)."""
    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
    target = np.ones_like(coords)
    target[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target


def keypoints_from_heatmaps(
    heatmaps,
    center,
    scale,
    post_process="unbiased",
    kernel=11,
    use_udp=False,
):
    """mmpose keypoints_from_heatmaps, GaussianHeatmap target type only.

    post_process: "unbiased" (DARK), "default" (+-0.25px shift), or None
    (plain argmax). use_udp switches both the refinement (post_dark_udp) and
    the back-projection convention.
    """
    heatmaps = heatmaps.copy()
    N, K, H, W = heatmaps.shape
    if use_udp:
        preds, maxvals = _get_max_preds(heatmaps)
        preds = post_dark_udp(preds, heatmaps, kernel=kernel)
    else:
        preds, maxvals = _get_max_preds(heatmaps)
        if post_process == "unbiased":
            heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
            for n in range(N):
                for k in range(K):
                    preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
        elif post_process is not None:
            for n in range(N):
                for k in range(K):
                    heatmap = heatmaps[n][k]
                    px = int(preds[n][k][0])
                    py = int(preds[n][k][1])
                    if 1 < px < W - 1 and 1 < py < H - 1:
                        diff = np.array(
                            [
                                heatmap[py][px + 1] - heatmap[py][px - 1],
                                heatmap[py + 1][px] - heatmap[py - 1][px],
                            ]
                        )
                        preds[n][k] += np.sign(diff) * 0.25
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp
        )
    return preds, maxvals


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_DECODE = dict(use_udp=False, post_process="unbiased", kernel=11)


def _flip_index(flip_pairs, num_kps):
    index = list(range(num_kps))
    for a, b in flip_pairs:
        index[a], index[b] = index[b], index[a]
    return index


class VitposeModel:
    """Torch backend. Load once, call predict() per crop.

    Args: ckpt, device, decode (DEFAULT_DECODE overrides), flip_tta
    (horizontal-flip TTA — never for chiral objects like tetra), flip_pairs,
    mask_threshold (None = checkpoint's value).
    """

    def __init__(
        self,
        ckpt,
        device="cuda",
        decode=None,
        flip_tta=False,
        flip_pairs=None,
        mask_threshold=None,
    ):
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        for key in ("model", "active_heads", "img_size"):
            if key not in payload:
                raise KeyError(
                    f"{ckpt} has no '{key}' — not a valve-vision joint "
                    "checkpoint (valve-era flat checkpoints are unsupported)"
                )
        state = payload["model"]
        train_config = payload.get("train_config", {})
        self._init_meta(
            active_heads=payload["active_heads"],
            img_size=payload["img_size"],
            mask_threshold=(
                mask_threshold
                if mask_threshold is not None
                else train_config.get("mask_threshold", 0.5)
            ),
            amp=train_config.get("amp", False),
        )

        embed_dim = int(state["backbone.patch_embed.proj.bias"].shape[0])
        if embed_dim not in _ARCHITECTURES:
            raise ValueError(f"unknown ViT embed_dim {embed_dim}")
        backbone_cfg = dict(
            _ARCHITECTURES[embed_dim],
            img_size=(self.img_h, self.img_w),
            patch_size=PATCH_SIZE,
            embed_dim=embed_dim,
            ratio=1,
            mlp_ratio=4,
            qkv_bias=True,
        )
        pose_cfg = _head_cfg_from_state(state, "pose_head", embed_dim)
        mask_cfg = (
            _head_cfg_from_state(state, "mask_head", embed_dim)
            if "seg" in self.active_heads
            else None
        )
        self._init_heads(
            pose_cfg["out_channels"],
            mask_cfg["out_channels"] if mask_cfg else 0,
            decode,
            flip_tta,
            flip_pairs,
        )
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.model = JointVitpose(backbone_cfg, pose_cfg, mask_cfg)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(
            f"VitposeModel loaded: {ckpt} (ViT dim {embed_dim}, "
            f"K={self.num_kps}, C={self.num_masks}, "
            f"input {self.img_h}x{self.img_w}, decode={self.decode}, "
            f"{self.device})"
        )

    # Shared with VitposeTRT: everything except how a tensor becomes heatmaps.
    def _init_meta(self, active_heads, img_size, mask_threshold, amp):
        self.active_heads = tuple(active_heads)
        self.img_h, self.img_w = (int(v) for v in img_size)
        self.mask_threshold = float(mask_threshold)
        self.amp = bool(amp)

    def _init_heads(self, num_kps, num_masks, decode, flip_tta, flip_pairs):
        self.num_kps = int(num_kps)
        self.num_masks = int(num_masks)
        self.decode = dict(DEFAULT_DECODE)
        self.decode.update(decode or {})
        self.flip_tta = bool(flip_tta)
        self.flip_pairs = [list(p) for p in (flip_pairs or [])]
        self._flip_index = _flip_index(self.flip_pairs, self.num_kps)

    # -------------------------------------------------------------- internals

    def _normalize(self, crop_rgb):
        arr = ((crop_rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(arr[None])).to(self.device)

    def _forward(self, tensor):
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=bool(self.amp and self.device.type == "cuda"),
        ):
            heatmaps, mask_logits = self.model(tensor)
        heatmaps = heatmaps.float()
        if mask_logits is not None:
            mask_logits = mask_logits.float()
        return heatmaps, mask_logits

    # ----------------------------------------------------------------- public

    @torch.no_grad()
    def predict(self, img_rgb, bbox_xywh):
        """One RGB crop -> (kps (K, 2), scores (K, 1), mask_probs
        (C, H_src, W_src) | None), everything in source-image coordinates."""
        center, scale = box2cs(bbox_xywh, self.img_w, self.img_h)
        transform = get_affine_transform(
            center, scale, PIXEL_STD, 0, (self.img_w, self.img_h)
        )
        crop = cv2.warpAffine(
            img_rgb, transform, (self.img_w, self.img_h), flags=cv2.INTER_LINEAR
        )
        heatmaps, mask_logits = self._forward(self._normalize(crop))

        if self.flip_tta:
            hm_f, ml_f = self._forward(self._normalize(crop[:, ::-1, :].copy()))
            hm_f = hm_f.flip(-1)[:, self._flip_index]
            heatmaps = (heatmaps + hm_f) * 0.5
            if mask_logits is not None and ml_f is not None:
                # No mask channel swaps for current objects; mirror spatially.
                mask_logits = (mask_logits + ml_f.flip(-1)) * 0.5

        kps, scores = keypoints_from_heatmaps(
            heatmaps.cpu().numpy(),
            center[None],
            (scale * PIXEL_STD)[None],
            post_process=self.decode["post_process"],
            kernel=int(self.decode["kernel"]),
            use_udp=bool(self.decode["use_udp"]),
        )

        mask_probs = None
        if mask_logits is not None:
            # Warp the CONTINUOUS probabilities to source resolution;
            # thresholding first aliases (§1.3 contract).
            probs_crop = (
                F.interpolate(
                    torch.sigmoid(mask_logits),
                    size=(self.img_h, self.img_w),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                .cpu()
                .numpy()
            )
            inverse = cv2.invertAffineTransform(transform)
            mask_probs = np.stack(
                [
                    cv2.warpAffine(
                        plane,
                        inverse,
                        (img_rgb.shape[1], img_rgb.shape[0]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    for plane in probs_crop
                ]
            )
        return kps[0], scores[0], mask_probs


# ══════════════════════════════════════════════════════════════════════════════
# TensorRT backends (.engine + sidecar .json, Orin)
# ══════════════════════════════════════════════════════════════════════════════
#
# Export chain (utils/vitpose_export.py -> utils/vitpose_build_engine.py):
#   ckpt.pth --> model.onnx + model.json (sidecar: kind, img_size, K, C,
#   mask_threshold, io names) --> model.engine (built ON the target GPU;
#   engines are device-specific). The sidecar is required: an engine carries
#   shapes but not the checkpoint metadata predict() needs.
#
# Design: torch owns the CUDA memory (torch is on the Orin already, and its
# primary context works from any rospy thread — the pycuda push/pop dance of
# the valve-era ValvePoseTRT is gone). TRT only gets raw device pointers and
# torch's current stream. Pre/post-processing is the torch classes' own code:
# VitposeTRT/ObjectnessTRT subclass them and override only _forward.
# TRT 8.x (Jetson JetPack 5) and 10.x (JetPack 6 / x86 pip) both supported.

_TRT_LOGGER = None


def _sidecar_path(engine_path):
    root, _ = os.path.splitext(str(engine_path))
    return root + ".json"


def load_sidecar(engine_path):
    """Metadata written next to the ONNX/engine by vitpose_export.py."""
    path = _sidecar_path(engine_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} missing — every .engine needs the sidecar .json written "
            "by utils/vitpose_export.py (copy it next to the engine)"
        )
    with open(path, "r") as handle:
        return json.load(handle)


class TrtEngine:
    """Minimal TensorRT executor over torch CUDA buffers.

    run(input_tensor) -> {output_name: torch tensor (fresh copy)}. Static
    shapes, batch 1. Buffers are allocated once from the engine's IO shapes.
    """

    def __init__(self, engine_path):
        import tensorrt as trt  # deferred: torch-only deployments never need it

        global _TRT_LOGGER
        if _TRT_LOGGER is None:
            _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"TRT engine not found: {engine_path}")
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend needs a CUDA device")
        self.trt = trt
        self.path = str(engine_path)
        with open(engine_path, "rb") as handle:
            blob = handle.read()
        runtime = trt.Runtime(_TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.device = torch.device("cuda")

        self.inputs = {}  # name -> tensor
        self.outputs = {}
        self._named_api = hasattr(self.engine, "num_io_tensors")  # TRT >= 8.5
        if self._named_api:
            names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            for name in names:
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype = _torch_dtype(trt, self.engine.get_tensor_dtype(name))
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs[name] = buf
                else:
                    self.outputs[name] = buf
                self.context.set_tensor_address(name, int(buf.data_ptr()))
        else:  # TRT 8.0-8.4 positional bindings
            self._bindings = []
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = tuple(self.engine.get_binding_shape(i))
                dtype = _torch_dtype(trt, self.engine.get_binding_dtype(i))
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                (self.inputs if self.engine.binding_is_input(i) else self.outputs)[
                    name
                ] = buf
                self._bindings.append(int(buf.data_ptr()))
        if len(self.inputs) != 1:
            raise RuntimeError(
                f"{engine_path}: expected exactly one input, got "
                f"{sorted(self.inputs)}"
            )
        self.input_name = next(iter(self.inputs))
        self.input_shape = tuple(self.inputs[self.input_name].shape)

    def run(self, tensor):
        buf = self.inputs[self.input_name]
        if tuple(tensor.shape) != tuple(buf.shape):
            raise ValueError(
                f"{self.path}: input {tuple(tensor.shape)} != engine "
                f"{tuple(buf.shape)}"
            )
        stream = torch.cuda.current_stream()
        buf.copy_(tensor.to(self.device, dtype=buf.dtype, non_blocking=True))
        if self._named_api:
            ok = self.context.execute_async_v3(stream.cuda_stream)
        else:
            ok = self.context.execute_async_v2(self._bindings, stream.cuda_stream)
        if not ok:
            raise RuntimeError(f"{self.path}: TensorRT execution failed")
        stream.synchronize()
        return {name: out.clone() for name, out in self.outputs.items()}


def _torch_dtype(trt, trt_dtype):
    table = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
    }
    if hasattr(trt.DataType, "BOOL"):
        table[trt.DataType.BOOL] = torch.bool
    return table[trt_dtype]


class VitposeTRT(VitposeModel):
    """TensorRT joint model — drop-in for VitposeModel (same constructor
    surface; `device` is ignored, TRT is CUDA). Reads <engine>.json for the
    checkpoint metadata; heatmaps/mask_logits come back as torch tensors and
    flow through VitposeModel.predict unchanged."""

    def __init__(
        self,
        engine,
        device="cuda",
        decode=None,
        flip_tta=False,
        flip_pairs=None,
        mask_threshold=None,
    ):
        meta = load_sidecar(engine)
        if meta.get("kind") != "joint":
            raise ValueError(f"{engine}: sidecar kind {meta.get('kind')!r} != 'joint'")
        self._init_meta(
            active_heads=meta["active_heads"],
            img_size=meta["img_size"],
            mask_threshold=(
                mask_threshold
                if mask_threshold is not None
                else meta.get("mask_threshold", 0.5)
            ),
            amp=False,
        )
        self._init_heads(
            meta["num_kps"], meta.get("num_masks", 0), decode, flip_tta, flip_pairs
        )
        self.engine = TrtEngine(engine)
        self.device = self.engine.device
        self.model = None
        expect = (1, 3, self.img_h, self.img_w)
        if self.engine.input_shape != expect:
            raise ValueError(
                f"{engine}: input {self.engine.input_shape} != sidecar {expect}"
            )
        self._out_heatmaps = meta["outputs"]["heatmaps"]
        self._out_masks = meta["outputs"].get("mask_logits")
        if self._out_masks and self._out_masks not in self.engine.outputs:
            raise ValueError(f"{engine}: output {self._out_masks!r} not in engine")
        print(
            f"VitposeTRT loaded: {engine} (K={self.num_kps}, C={self.num_masks}, "
            f"input {self.img_h}x{self.img_w}, decode={self.decode})"
        )

    def _forward(self, tensor):
        outs = self.engine.run(tensor)
        heatmaps = outs[self._out_heatmaps].float()
        mask_logits = outs[self._out_masks].float() if self._out_masks else None
        return heatmaps, mask_logits


def load_vitpose(ckpt, **kwargs):
    """Backend-agnostic loader: .engine -> VitposeTRT, else VitposeModel."""
    if str(ckpt).endswith(".engine"):
        return VitposeTRT(ckpt, **kwargs)
    return VitposeModel(ckpt, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Objectness: full-frame box detector (gate_tetra_overview.md §4)
# ══════════════════════════════════════════════════════════════════════════════
#
# Same trunk (ViT-S), no deconvolutions: a 3x3 + 1x1 head on the native
# stride-16 grid, one logit per patch = "how much of this patch is object".

OBJECTNESS_STRIDE = 16
SOURCE_ASPECT = 4.0 / 3.0


class ObjectnessNet(nn.Module):
    """ViT trunk + stride-16 coverage head (attribute names match the ckpt)."""

    def __init__(self, backbone_cfg, hidden=256):
        super().__init__()
        self.backbone = ViT(**backbone_cfg)
        channels = backbone_cfg["embed_dim"]
        # Index 3 is an Identity standing in for the training-time Dropout2d,
        # so the classifier stays at head.4 and the state dict loads as saved.
        self.head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):
        """(B, 3, H, W) -> (B, 1, H/16, W/16) logits."""
        return self.head(self.backbone(x))


def letterless_crop(image):
    """Centre-crop to the training renders' 4:3; returns (crop, (ox, oy)).

    Squashing would change the object's aspect and letterboxing adds a border
    training never shows (measurements: gate_tetra_overview.md §4.2). A 4:3
    input passes through untouched.
    """
    h, w = image.shape[:2]
    if w / float(h) > SOURCE_ASPECT:
        cw, ch = int(round(h * SOURCE_ASPECT)), h
    else:
        cw, ch = w, int(round(w / SOURCE_ASPECT))
    ox, oy = (w - cw) // 2, (h - ch) // 2
    return image[oy : oy + ch, ox : ox + cw], (float(ox), float(oy))


def decode_box(prob, threshold=0.5, stride=OBJECTNESS_STRIDE, measure_threshold=None):
    """Largest above-threshold blob of a coverage map -> (bbox_xywh, score).

    Upsample to input resolution BEFORE thresholding — that recovers sub-cell
    precision from coverage-fraction training. `measure_threshold` cuts the
    box EXTENT tighter than detection (real-footage ramps are wider than
    sim's; measurements: gate_tetra_overview.md §4.3); if nothing survives
    it, the detection-cut box stands. Coordinates in model-input pixels.
    """
    grid_h, grid_w = prob.shape
    full = cv2.resize(
        prob, (grid_w * stride, grid_h * stride), interpolation=cv2.INTER_LINEAR
    )
    mask = (full >= threshold).astype(np.uint8)
    if not mask.any():
        return None, float(prob.max())
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    score = float(full[labels == best].max())
    if measure_threshold is not None:
        inner = (labels == best) & (full >= measure_threshold)
        if inner.any():
            ys, xs = np.nonzero(inner)
            return [
                float(xs.min()),
                float(ys.min()),
                float(xs.max() - xs.min() + 1),
                float(ys.max() - ys.min() + 1),
            ], score
    x, y, w, h = (
        stats[best, cv2.CC_STAT_LEFT],
        stats[best, cv2.CC_STAT_TOP],
        stats[best, cv2.CC_STAT_WIDTH],
        stats[best, cv2.CC_STAT_HEIGHT],
    )
    return [float(x), float(y), float(w), float(h)], score


class ObjectnessDetector:
    """Torch backend for the objectness checkpoint. Load once, predict per
    frame. Args: ckpt, device, threshold (detect), measure_threshold (extent
    only; None disables the split — see decode_box)."""

    def __init__(
        self,
        ckpt,
        device="cuda",
        threshold=0.5,
        measure_threshold=0.7,
    ):
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
        for key in ("model", "img_size"):
            if key not in payload:
                raise KeyError(
                    f"{ckpt} has no '{key}' — not a valve-vision objectness "
                    "checkpoint"
                )
        state = payload["model"]
        self.img_h, self.img_w = tuple(payload["img_size"])
        self.stride = int(payload.get("stride", OBJECTNESS_STRIDE))
        self.threshold = float(threshold)
        self.measure_threshold = (
            None if measure_threshold is None else float(measure_threshold)
        )

        embed_dim = int(state["backbone.patch_embed.proj.bias"].shape[0])
        if embed_dim not in _ARCHITECTURES:
            raise ValueError(f"unknown ViT embed_dim {embed_dim}")
        backbone_cfg = dict(
            _ARCHITECTURES[embed_dim],
            img_size=(self.img_h, self.img_w),
            patch_size=PATCH_SIZE,
            embed_dim=embed_dim,
            ratio=1,
            mlp_ratio=4,
            qkv_bias=True,
        )
        hidden = int(state["head.0.weight"].shape[0])
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.model = ObjectnessNet(backbone_cfg, hidden=hidden)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(
            f"ObjectnessDetector loaded: {ckpt} (ViT dim {embed_dim}, "
            f"input {self.img_h}x{self.img_w}, stride {self.stride}, "
            f"detect {self.threshold} / measure {self.measure_threshold}, "
            f"{self.device})"
        )

    def _forward(self, tensor):
        return self.model(tensor)

    @torch.no_grad()
    def predict(self, img_rgb, return_prob=False):
        """Full RGB frame -> (bbox_xywh, score) in source pixels, bbox None
        if nothing fired; (bbox, score, prob) when return_prob."""
        crop, (ox, oy) = letterless_crop(img_rgb)
        ch, cw = crop.shape[:2]
        resized = cv2.resize(
            crop, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA
        )
        normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(
            np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])
        ).to(self.device)
        prob = torch.sigmoid(self._forward(tensor).float()).cpu().numpy()[0, 0]
        box, score = decode_box(
            prob,
            self.threshold,
            stride=self.stride,
            measure_threshold=self.measure_threshold,
        )
        if box is not None:
            sx, sy = cw / float(self.img_w), ch / float(self.img_h)
            box = [
                box[0] * sx + ox,
                box[1] * sy + oy,
                box[2] * sx,
                box[3] * sy,
            ]
        return (box, score, prob) if return_prob else (box, score)


class ObjectnessTRT(ObjectnessDetector):
    """TensorRT objectness detector — drop-in for ObjectnessDetector."""

    def __init__(self, engine, device="cuda", threshold=0.5, measure_threshold=0.7):
        meta = load_sidecar(engine)
        if meta.get("kind") != "objectness":
            raise ValueError(
                f"{engine}: sidecar kind {meta.get('kind')!r} != 'objectness'"
            )
        self.img_h, self.img_w = (int(v) for v in meta["img_size"])
        self.stride = int(meta.get("stride", OBJECTNESS_STRIDE))
        self.threshold = float(threshold)
        self.measure_threshold = (
            None if measure_threshold is None else float(measure_threshold)
        )
        self.engine = TrtEngine(engine)
        self.device = self.engine.device
        self.model = None
        expect = (1, 3, self.img_h, self.img_w)
        if self.engine.input_shape != expect:
            raise ValueError(
                f"{engine}: input {self.engine.input_shape} != sidecar {expect}"
            )
        self._out_logits = meta["outputs"]["logits"]
        print(
            f"ObjectnessTRT loaded: {engine} (input {self.img_h}x{self.img_w}, "
            f"stride {self.stride}, detect {self.threshold} / measure "
            f"{self.measure_threshold})"
        )

    def _forward(self, tensor):
        return self.engine.run(tensor)[self._out_logits]


def load_objectness(ckpt, **kwargs):
    """Backend-agnostic loader: .engine -> ObjectnessTRT, else ObjectnessDetector."""
    if str(ckpt).endswith(".engine"):
        return ObjectnessTRT(ckpt, **kwargs)
    return ObjectnessDetector(ckpt, **kwargs)

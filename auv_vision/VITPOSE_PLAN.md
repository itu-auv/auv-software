# ViTPose gate/tetra pipeline — decisions & verification record

TEKNOFEST "yıldızlar": gate (9 kp + aperture mask) and tetra (7 kp + 3
colour-face masks), models trained in valve-vision. This file holds the
locked design decisions and the verification evidence. The model contract
(preprocess, decode, objectness) is `gate_tetra_overview.md` at the repo
root; the operational map is `auv_vision/CLAUDE.md`; mechanism rationale
lives with the code (section banners in `utils/vitpose_utils.py`).

## Decisions locked

- Valve support removed entirely. `VitposeResult` is new, zero backwards
  compatibility; keypoint ids 0-indexed everywhere.
- One detection-node instance runs one model at a time (switchable live via
  `~set_config`); simultaneous objects = multiple instances.
- Input resolution is the checkpoint's own — the pos-embed is trained at
  that size; a different resolution means training a model at it.
- Checkpoints: gate/010, tetra/004. 004's val numbers trail 003's slightly
  because 004 trained WITH random chirality — 003 would fail on
  mirror-arranged tetras.
- TensorRT: `.engine` files load via `VitposeTRT`/`ObjectnessTRT` (torch-owned
  CUDA buffers, no pycuda). Chain: `utils/vitpose_export.py` (pth → onnx +
  sidecar json, onnxruntime parity ≤1e-5 on all four models, 08-17) →
  `utils/vitpose_build_engine.py` on the target GPU → `utils/vitpose_trt_check.py`
  parity + timing. Engine runtime path unverified on hardware until the Orin.
- No objectness fallback: a frame with no detection publishes no result
  (full-frame joint inference on real footage is measurably useless).
- Poses go through the object map TF server, never raw TF broadcasts.
  Calibration via `CameraCalibrationFetcher`. Node state = one mode
  (`off`/`detect`/`pose`, `~set_mode` SetString; `~enable` = fullest/off
  alias) with loaded ≡ enabled: leaving a mode frees its models from the
  GPU (torch's CUDA context, ~225 MiB, stays until process exit). Status
  latched: `~enabled` Bool + `~mode` String. Debug topics subscriber-gated.

## Components

- `auv_msgs/VitposeResult.msg` — keypoints + mono8 probability masks at
  source resolution + calibrated threshold.
- `scripts/utils/vitpose_inference.py` — torch runtimes (joint + objectness),
  faithful valve-vision ports; parity is the acceptance test.
- `scripts/utils/vitpose_utils.py` — ROS-free: config/validation, planar pose
  fusion, tetra association, crop tracking.
- `scripts/vitpose_detection_node.py` — producer: bbox providers, lazy
  loading, detect-only mode, and `VitposeNodeBase` (the service shell shared
  with sim).
- `scripts/sim_vitpose_node.py` — ground-truth twin, same interface; geometry
  from `config/sim_vitpose_objects.yaml`, no checkpoints touched.
- `scripts/vitpose_process_node.py` + `scripts/vitpose_ops/` — consumer; ops
  `gate_pose` and `tetra_unfold`; debug + idle overlay.
- `config/vitpose/<object>.yaml` — schema: `object` + `camera` + `detection`
  required, `model` + `process` optional; no `model:` section = detect-only.
  `camera: front|bottom|torpedo` expands at load into image/result topics,
  calibration ns and optical frame (`apply_camera`, standard cam layout).
- `auv_bringup/launch/yildiz.launch` — pose node + detect-only front scan
  node + process node; `sim:=true` swaps node type, never names.
- `utils/slim_checkpoint.py` — strips a joint training checkpoint to its
  inference payload (1.13 GB → ~380 MB); objectness ships as trained.

## Verification record

Harnesses live on dream (`~`) / in the ROS container home. Dates 2026-08-09
to 08-14.

| what | harness | result |
|---|---|---|
| decode parity vs `tools/infer_joint.py` | offline (§9.1-era) | keypoints float-exact; UDP decode degrades ~3× (7.38 vs 2.38 px) |
| gate fused pose | `~/gate_fusion_verify` | clean = label-noise floor; occlusion (4 near-collinear kps): kp-only 4.7°/10 cm median, worst 30° → fused 2°/4 cm, worst 6°; ~14 ms CPU |
| tetra association | `~/tetra_unfold_check` | 100/100 single-frame; locks in 4 frames; 100% correct up to ρ=0.7 letter-swap noise; averted letter recovered 100% and flagged `inferred`; τ=10/clip=20 and lock 0.99/10 chosen by sweep |
| objectness parity | `~/objectness_parity` | boxes identical to 0.0 px on 24 images, coverage maps to 6e-5 |
| real-footage chain | `~/tetra_e2e` | fires 74/80 (all 6 misses = object leaving the 4:3 crop); face masks + vertices land correctly; association correctly UNCERTAIN — that prop has no letters, so letter association is verified on sim val only |
| crop tracking | `~/tetra_e2e/validate_tracker.py` | output IoU flat in seed period (0.61 at 1–4 s), 75–84% detector calls saved; beats VitTrack (0.72 vs 0.61) at zero cost; live: 5% vs 28% GPU at 10 Hz |
| gate objectness | `~/yildiz_smoke` | fires 100/100 on gate val at median score 1.000; the old tetra placeholder fired 20/100 — false boxes, worse than silence |
| detect-only + lazy load | `~/yildiz_smoke` | 0.94 Hz at rate 1.0, enable 1.15 s (920→1339 MiB), heartbeat present, all guardrails reject with reasons, failure path leaves the node up |
| shell refactor smoke | live, 2026-08-14 | 9.3 Hz results, live gate→tetra swap, strict sim swap, warm-up heartbeats at the rate cap, re-enable 863 ms |
| mode interface | `~/yildiz_smoke/mode_smoke.py`, live 2026-08-15 | full off/detect/pose matrix green, real + sim; transitions ~1.3 s; VRAM 38 cold → 457 detect → 877 pose → 513 detect → 225 off (CUDA-context floor); detect-only clamp + pose refusal verified |
| timing (4060 Ti, 640×480) | — | objectness 27.1 ms (**82% of GPU**), joint 6.7 ms → 29.6 Hz; 1080p source balloons mask decode 0.8→13.3 ms; Orin estimate ~7 Hz per-frame / ~30 Hz seeded |

Open caveats:

- Letter association unverified on real lettered footage.
- Gate `measure_threshold` deliberately null until real gate footage exists
  to measure against.
- Tracker lag numbers come from a 2.8 Hz clip (worst case); re-validate on
  high-rate footage.
- Jetson/Orin timing untested; TRT engines not yet built/verified on the Orin
  — first thing to check if it must run onboard.

## Landmines (paid for once)

- Offline harnesses exercised the utils; the K=3→7 tetra retrain crashed the
  op on every frame and only the live ROS smoke caught it. Test the
  integration seam (the node/op), not just the library.
- "Silhouette touches the border" checks must test the box2cs crop
  (aspect-fitted, 1.25× padded), not the bbox — testing the bbox silently
  defeated the tracker (10% saved instead of 75%).
- `solvePnPRansac` forces EPnP and breaks on coplanar points; use
  `solvePnPGeneric`/IPPE.

## Out of scope

- A tetra pose op from the 4 vertex keypoints (predicted, unconsumed).
- A 16:9 objectness model (today the 4:3 centre crop blinds 25% of
  cam_bottom's horizontal FOV).
- SMACH states, `robosub.launch` wiring. Any valve resurrection.

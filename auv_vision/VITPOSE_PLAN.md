# ViTPose gate/tetra pipeline — implementation plan

Branch: `ufuk/yıldızlar-ikinci-görev` (off `main`; nothing is merged from
`last_dance2` wholesale — the three valve files are used as raw material only).
Mission: TEKNOFEST "yıldızlar" — gate (9 kp + aperture mask) and tetra
(3 letter kps + 3 colour-face masks), models trained in valve-vision.
Normative model contract: `gate_tetra_overview.md` (repo root, untracked).

> **Status 2026-08-09: scaffolding pass SHIPPED** on this branch (msgs →
> runtime → nodes → configs, 3 commits). Parity + smoke evidence in the
> commit messages. Everything under "Future ops" remains unimplemented.

## Scope of THIS pass: scaffolding

Everything below section 5's "Future ops" heading is **design sketch, not
current work**. The deliverable of this pass, end to end:

- feed images + calibrations in,
- get inference **debug images** out (keypoints, skeleton, masks, bbox),
- and have a **stub op** in the process node be called per frame with real
  `FrameData` — proving the op seam works.

No gate_pose, no tetra_unfold, no pnp_utils. The framework must make it obvious
where those plug in, nothing more.

## 0. Decisions locked

- Valve support is **removed entirely** (no legacy checkpoint loading, no
  valve YAML, no flip-index constants, no VITTrack).
- New message `VitposeResult`; **zero backwards compatibility** concerns.
  `KeypointResult` is left untouched for whatever still uses it.
- Keypoint ids are **0-indexed** everywhere, matching valve-vision.
- TensorRT backend is a **stub** until an engine is actually exported.
- One detection-node instance runs **one model at a time** (runtime-switchable);
  simultaneous objects = multiple instances.
- Checkpoints: **gate/010** and **tetra/004**. 004's val metrics are a hair
  behind 003 (pose 2.03 vs 1.82 px source, mask IoU 0.969 vs 0.974) precisely
  because 004 trained with **random chirality** (the red↔green surface swap) —
  003's cleaner numbers come from the easier fixed-arrangement task and it
  would fail on mirror-arranged tetras. 001/002 were a broken run and a
  small-data smoke run.

## 1. Models & deployment

- Slim-export script `auv_vision/scripts/utils/slim_checkpoint.py` (committed;
  reusable after every retrain): loads a valve-vision joint checkpoint, keeps
  only `{model, active_heads, model_size, img_size, train_config}`, drops
  optimizer/scheduler/scaler/metrics/manifest → 1.13 GB → ~380 MB.
- Deployed to `auv_detection/models/gate_joint.pth`, `tetra_joint.pth`
  (`*.pth` already gitignored; models ride the filesystem, house style).
- The slim payload keeps the full contract keys, so the runtime auto-configures
  from the checkpoint (K, C, B/L, input size, mask_threshold) with YAML overrides.

## 2. New message: `auv_msgs/VitposeResult.msg`

```
# One inference result: keypoints + segmentation for a single object instance.
std_msgs/Header header
string object                 # config object name, e.g. "gate", "tetra"
float64[] bbox                # [x, y, w, h] px fed to the model; empty = full frame
auv_msgs/Keypoint[] keypoints # 0-indexed ids, source-image px, raw confidences
string[] keypoint_names       # index-aligned with keypoint ids
string[] mask_classes         # profile declaration order, e.g. [red, green, blue]
sensor_msgs/Image[] masks     # mono8 probability maps (0..255 = 0..1),
                              # source resolution, index-aligned with mask_classes
float32 mask_threshold        # calibrated binarization threshold (from checkpoint)
```

Choices: probability maps, not binary — consumers threshold themselves (or use
soft coverage); the calibrated threshold rides along. Masks are already
inverse-warped to source resolution by the producer (per the §1.3 contract:
warp probabilities, *then* threshold — binarizing in crop space aliases).
`Keypoint.msg` stays as it is on main (`id/x/y/confidence`); the per-message
`object` field makes a per-keypoint object tag redundant.

## 3. `scripts/utils/vitpose_inference.py` — generalized runtime

Self-contained (torch/numpy/cv2 only), rewritten around the JointModel contract.

**`VitposeModel`** (torch backend):
- Loads the joint payload: `state = ck["model"]` with `backbone.* / pose_head.*
  / mask_head.*` prefixes. Module attribute names match the checkpoint so the
  state dict loads without remapping. `active_heads` decides which heads exist
  (`("pose",)`-only checkpoints get no mask head).
- Auto-config from checkpoint: embed dim → ViT-B/L, K from
  `pose_head.final_layer.bias`, C from `mask_head.final_layer.bias`,
  `(H, W)` from `img_size`, `mask_threshold` from `train_config`. Every value
  overridable from the YAML `model:` section. No module-level size constants;
  resolution is instance state (any 16-divisible input works — patch grid and
  pos-embed sizes derive from it).
- **Preprocess** (exact §1.1 contract): `box2cs` with aspect from configured
  input size, `PIXEL_STD=200`, fixed 1.25 pad → affine warp → RGB,
  ImageNet normalize.
- **Decode, config-driven** (`model.decode`):
  - `use_udp` (bool, default **false**) — controls the `transform_preds`
    rescale (`scale/size` vs `scale/(size-1)`). The valve code silently did
    UDP-style rescale; the gate/tetra heads are MSRA-encoded and measured
    3× worse under UDP decode, so non-UDP is the default and UDP is one flag.
  - `post_process`: `"unbiased"` (default; gaussian-blur→log→Taylor, i.e.
    DARK) | `"default"` | `null` (plain argmax + 0.25px shift).
  - `kernel` (default 11).
  - Ported 1:1 from valve-vision's decode path so offline parity is exact.
- **Mask decode** (§1.3): sigmoid → bilinear to crop size → `invertAffineTransform`
  → warp probabilities to source resolution. Returns probs; thresholding is the
  caller's job.
- **Flip TTA**: only if config provides `flip_pairs` (gate
  `[[0,1],[2,3],[5,7]]`); tetra has none → TTA structurally off. Flips both
  heatmaps and mask logits (mask channels have no swap pairs for our objects;
  mask TTA simply mirrors spatially).
- `predict(img_rgb, bbox_xywh) -> (kps (K,2), scores (K,1), mask_probs (C,H,W) | None)`
  — everything in source-image coordinates.

**`VitposeTRT`**: stub class raising `NotImplementedError` with a docstring
describing the intended two-output binding (pose + mask tensors) and the
pycuda push/pop-per-call pattern proven on the valve TRT path.
**`load_vitpose(path, cfg)`** dispatches on extension (`.engine` → stub).

## 4. `scripts/vitpose_detection_node.py` — producer

Deliberately thin: image in → (bbox source) → model → `VitposeResult` out.
All tracker logic, state machine, seed workers and the RANSAC plausibility
gate from the valve node are gone (pose-quality judgment moves to process ops).

- **`BboxProvider`** interface — the objectness seam:
  `get_bboxes(img_bgr, header) -> List[bbox_xywh]`.
  - `full_frame` (default): one bbox covering the image — the "assume already
    cropped" mode.
  - `topic`: subscribes `vision_msgs/Detection2DArray` (house format — the
    realsense YOLO bridge already emits it), optional class-id filter, staleness
    gate. A future objectness/tracker node only needs to publish this topic.
  - Provider chosen by `detection.bbox_provider.type`; adding a new provider =
    one subclass + one config value.
- Per image callback: BGR→RGB once, run `predict()` per bbox (typically 1),
  publish one `VitposeResult` per bbox. `min_kp_conf_to_publish` optionally
  zero-suppresses nothing — keypoints are published unfiltered with raw
  confidences (house style: consumer applies its own gate; debug shows all).
- **Runtime model switching**: `~set_config` service
  (`auv_msgs/SetObjectTransform`-style string request → implemented with a small
  new srv `SetString` if none fits; check `auv_msgs/srv` first). Accepts an
  object name (resolved to `config/vitpose/<name>.yaml`) or an absolute path.
  Loads config + checkpoint on a worker thread, swaps atomically under a lock;
  frames keep flowing through the old model until the swap. Publishes nothing
  during the brief swap itself.
- **Enable convention** (mirrors the YOLO tracker nodes so the
  camera_detection-style 1 Hz sync could manage GPU load later): `~enabled`
  param, `~enable` SetBool service, `~enabled` Bool status published at 1 Hz.
- Private (`~`) services throughout so multiple instances coexist.

## 5. `scripts/vitpose_process_node.py` — operation framework

Consumes `VitposeResult`, runs configured **operations** per object, follows
the codebase's dynamic-import handler pattern.

- Loads **every** YAML in `config/vitpose/` at startup and builds ops per
  object; incoming messages dispatch on `msg.object`. Consequence: the process
  node never needs config switching — only the detection node (which holds the
  single GPU model) switches. New objects are picked up by restart.
- **Ops as modules**: `scripts/vitpose_ops/<type>.py`, each exporting
  `create_op(params, ctx)` (same factory convention as `scripts/handlers/`).
  `process.operations: [{type: gate_pose, params: {...}}]` → importlib.
- **`OpContext`** (what an op gets at setup): camera frame id, `K`/`D` (via
  `auv_common_lib` `CameraCalibrationFetcher` — house style, not a raw
  camera_info subscription), tf_buffer, and helpers:
  - `publish_tf(child_frame_id, R, t, stamp)` → `TransformStamped` on
    `object_transform_updates` via `transform_to_odom_and_publish(...,
    rotation_quat=...)` — i.e. **through the object map TF server** (remapped
    to `map/object_transform_updates` in launch), never a raw TF broadcast.
    The `rotation_quat` extension to `transform_to_odom_and_publish` is ported
    from last_dance2 (the only detection_utils change carried over, plus the
    shape-factory/`build_pose_keypoints` helpers).
  - `publisher(topic, msg_type)` → lazily-created `rospy.Publisher` for
    op-specific outputs.
  - `draw(fn)` → registers an overlay callback for the debug image.
- **`FrameData`** (what an op gets per message): ids, pixels (N,2), scores,
  mask_probs (C,H,W float 0..1), mask_classes, mask_threshold, bbox, stamp.
- **Debug overlay** per camera: `vitpose_process_image_<cam>/compressed`
  (CompressedImage, JPEG q80), stamp-matched raw-image ring buffer, all work
  gated on `get_num_connections() > 0` (house style). Base layer draws
  keypoints (conf-coloured), skeleton from config, bbox, and translucent mask
  contours; ops add their own layers via `draw()`.
### This pass ships exactly one op: `stub`

`vitpose_ops/stub.py` — receives `FrameData`, logs a throttled one-line summary
(kp count/mean score, mask coverage per class), and registers a trivial
`draw()` callback (its name in a corner) to prove the overlay hook. That's it.
It doubles as the documented template for writing real ops.

### Future ops — design sketches only, NOT this pass

These are kept because the designs are worth keeping, not because they're
scheduled. When their time comes, `utils/pnp_utils.py` gets the `PnPEstimator`
carried over from `keypoint_pose_node` with its machinery intact:
`solvePnPGeneric` (exposes both coplanar IPPE candidates; `solvePnPRansac` is
unusable — it forces EPnP which breaks on coplanar points), reprojection inlier
gating, inlier re-solve through the flip-aware picker, and the **IPPE
flip-ambiguity guard** (near-equal reprojection errors → prefer the candidate
whose plane normal agrees with the last accepted one). Handle-line code is
dropped (valve-specific).

**`gate_pose` (sketch)** (uses keypoints *and* the mask):
1. Gate confident kps (config threshold), require `min_keypoints` (default 5).
2. Planar PnP via `PnPEstimator` (solver `ippe`, model points from config —
   §2.2 geometry, all 9 coplanar in the gate's y=0 plane; flip guard active).
   Model X = gate normal, consistent with the doc's frame.
3. **Mask consistency check**: project the aperture polygon (x∈[−0.5,0.5],
   z∈[0.06,1.46]) through the solved pose; compare against the thresholded
   aperture mask (IoU in source px). Below `min_mask_iou` (default ~0.4) the
   solve is rejected — this is the cheap "very reliable pose" validator that
   catches mirror-flip and outlier-driven solves that reproject fine on 5 points
   but put the aperture in the wrong place. Mask absent/empty → check abstains.
4. Known error profile guard: post midpoints (ids 5, 7) dominate the tail —
   they are listed in config as `soft_ids` and get a laxer inlier threshold
   rather than poisoning the solve.
5. Publish configured outputs (child frame + model-frame offset), e.g.
   `gate_link` at origin and `gate_entrance_link` at the aperture centre
   (0, 0, 0.76) — final naming per config, bare `_link` names per house rules.
   Distance gate via `max_distance` (as before).

**`tetra_unfold` (sketch)** (uses masks *and* keypoints; no PnP — tetra
keypoints are per-image glyph centres, not fixed 3D points):
1. For each letter kp above its score gate: letter → colour by
   **keypoint-in-mask** lookup on thresholded masks; fallback = nearest mask
   within `max_mask_distance_px`; unmatched/low-score letters stay unassigned
   (predictions for a letter on a fully averted face are unconstrained garbage
   — the doc is explicit; the score gate is the only defense).
2. Consistency: at most one letter per colour; conflicts resolved by score,
   loser unassigned. If exactly two letters are confidently assigned, the third
   association is **inferred** (the letter permutation is exhaustive) and
   flagged as inferred.
3. Publish association on `tetra/letter_colors` (`std_msgs/String`,
   e.g. `"A:red B:blue C:green(inferred)"` — String-topic precedent:
   `octagon/object_list`). Chirality is never assumed (50% mirror worlds);
   association comes only from the per-frame lookup.
4. Debug: an **unfolded-net image** — the canonical tetra net (3 colour
   triangles around the base) with detected letters drawn on their associated
   faces — published as `tetra_unfold_image/compressed` and mirrored into the
   overlay via `draw()`.

## 6. Config — one YAML per object, three sections

`auv_vision/config/vitpose/gate.yaml` (annotated; tetra analogous):

```yaml
object: gate

model:            # consumed by vitpose_inference (via the detection node)
  checkpoint: gate_joint.pth        # resolved against auv_detection/models/
  device: cuda                      # cuda | cpu
  input_size: null                  # [H, W]; null = from checkpoint (256, 192)
  decode: {use_udp: false, post_process: unbiased, kernel: 11}
  flip_tta: true
  flip_pairs: [[0, 1], [2, 3], [5, 7]]
  keypoint_names: [FrameTL, FrameTR, FrameBR, FrameBL,
                   MidTop, MidRight, MidBottom, MidLeft, PingerTop]
  mask_classes: [gate]              # declaration order = channel order
  mask_threshold: null              # null = from checkpoint (0.5)

detection:        # consumed by vitpose_detection_node
  image_topic: /taluy/cameras/cam_front/image_raw
  result_topic: /vitpose_result_front
  bbox_provider: {type: full_frame}   # future: {type: topic, topic: ..., class_id: ...}

process:          # consumed by vitpose_process_node
  camera:
    frame: taluy/base_link/front_camera_optical_link
    calibration_ns: cameras/cam_front   # CameraCalibrationFetcher namespace
  result_topic: /vitpose_result_front
  image_topic: /taluy/cameras/cam_front/image_raw   # debug overlay only
  skeleton: [[0,4],[4,1],[1,5],[5,2],[2,6],[6,3],[3,7],[7,0],[4,8]]
  operations:
    - type: stub          # THIS PASS. Future: gate_pose with model_points,
      params: {}          # solver, thresholds, outputs — see the sketches in §5.
```

Future `model_points` will use the last_dance2 shape-factory spec
(`build_pose_keypoints`) — points, plus `circle`/`rect` free for other objects;
0-indexed ids like everything else (the valve YAML's 1-indexing dies with it).

`tetra.yaml` deltas: bottom camera topics, no `flip_pairs`/`flip_tta: false`,
`mask_classes: [red, green, blue]`, `keypoint_names: [A, B, C]`, same `stub`
op, no skeleton.

## 7. Launch

`auv_vision/launch/vitpose.launch`: args `object` (default `gate`, picks the
config), `namespace`, `device`; starts one detection node + one process node
under `ns=$(arg namespace)` with `object_transform_updates` remapped to
`map/object_transform_updates`. The object map server is *not* launched here
(it comes from `auv_mapping/launch/start.launch` as usual); an optional
`standalone:=true` arg adds it for bench testing, mirroring the old
valve launch's convenience includes.

## 8. Codebase conventions being followed

- TFs go to the **object map server topic**, never straight `tf` broadcasts.
- Calibration via `auv_common_lib` `CameraCalibrationFetcher`.
- Config-driven everything; `~config_file`-style params with package-relative
  defaults; dynamic import + `create_*` factory for pluggable parts.
- Enable = SetBool service + Bool status topic (tracker-node pattern).
- Debug images: `<name>_image_<cam>/compressed`, work gated on subscriber count.
- Per-unit init failures logged and contained (bad op config disables that op,
  not the node).
- Scripts are devel-space (not in `catkin_install_python`), like their peers.
- `KINETO_DISABLED=1` before torch import (PyTorch clock bug, house workaround).

## 9. Validation (before any ROS wiring is trusted)

1. **Offline parity test** (the empirical gate): pull a val image + GT from
   `dream:~/valve-vision/data/gate_joint_1000/val` (and tetra equivalent), run
   valve-vision's `tools/infer_joint.py` for reference keypoints/masks, run our
   `vitpose_inference.py` on the same image + bbox → keypoints must match to
   float tolerance, masks near-identical (known quirk: reference JSON's
   `mask_png` filename is wrong for C>1 — resolve per-class files). Also assert
   the decode-config knobs move the result as documented (UDP decode degrades
   gate val error ~3×).
2. **ROS smoke on dream** (GPU): publish val images via `image_publisher` /
   webcam, run both nodes, verify `VitposeResult` content, debug overlay
   images (kps + masks drawn correctly on gate AND tetra), the stub op's log
   line firing per frame, and the `~set_config` gate→tetra live switch.

(Future, with real ops: op unit tests — gate_pose pose recovery vs GT +
mask-IoU rejection under kp corruption; tetra_unfold association vs GT.)

## 10. Work order

1. Branch; port msg (`VitposeResult`) + `detection_utils` `rotation_quat`
   extension; `catkin build auv_software`.
2. `slim_checkpoint.py`; export + place both models.
3. `vitpose_inference.py`; offline parity test (§9.1).
4. `vitpose_detection_node.py` (+ providers, set_config).
5. `vitpose_process_node.py` framework + base overlay + `stub` op.
6. Configs + launch; end-to-end smoke (§9.2).
7. Commit sequence mirrors this order (msg → runtime → nodes → configs).

## 11. Out of scope (explicit)

- **`gate_pose` / `tetra_unfold` implementations** (sketches in §5 only).
- `pnp_utils.py` port and the shape-factory/`build_pose_keypoints` helpers
  (come with the first real PnP op).
- TRT engine export & backend (stub only).
- Objectness/tracker bbox provider (seam exists: Detection2DArray topic).
- SMACH integration, mission states, `robosub.launch` wiring.
- Any valve resurrection.

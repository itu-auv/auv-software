# ViTPose gate/tetra pipeline — implementation plan

Branch: `ufuk/yıldızlar-ikinci-görev` (off `main`; nothing is merged from
`last_dance2` wholesale — the three valve files are used as raw material only).
Mission: TEKNOFEST "yıldızlar" — gate (9 kp + aperture mask) and tetra
(3 letter kps + 3 colour-face masks), models trained in valve-vision.
Normative model contract: `gate_tetra_overview.md` (repo root, untracked).

> **Status 2026-08-09: scaffolding pass SHIPPED** on this branch (msgs →
> runtime → nodes → configs, 3 commits). Parity + smoke evidence in the
> commit messages.
>
> **Update, same day: `gate_pose` SHIPPED** (`vitpose_ops/gate_pose.py` +
> `utils/pnp_utils.py` + gate.yaml op config) — implemented as a *fused*
> estimator (design upgraded from the PnP-then-verify sketch below; the mask
> is a measurement source, not just a validator). Verified offline against
> pseudo-GT on gate_joint_1000 val (harness: dream:~/gate_fusion_verify):
> clean = label-noise floor; occlusion workload (4 surviving collinear kps,
> mask bottom corrupted) kp-only 4.7°/10 cm median (worst 30°) → fused
> 2°/4 cm (worst 6°), ~14 ms/frame CPU.
>
> **Update 2026-08-10: `tetra_unfold` SHIPPED, and the utils files were
> consolidated.** All shared vitpose code now lives in ONE module,
> `utils/vitpose_utils.py`, in three banner-separated sections (config /
> planar pose fusion / tetra association) — `pnp_utils.py` and
> `vitpose_config.py` are gone, and by standing preference (Ufuk, 2026-08-10)
> no new utils file is created per feature. The `stub` op is deleted; its
> op-writing template lives in `vitpose_process_node.py`'s docstring.
> Both real ops (`gate_pose`, `tetra_unfold`) now ship. Verification for
> tetra is in §5 and the harness at dream:~/tetra_unfold_check.

## Scope of the scaffolding pass (historical — both ops have since shipped)

The first pass deliberately shipped no ops. Its deliverable, end to end:

- feed images + calibrations in,
- get inference **debug images** out (keypoints, skeleton, masks, bbox),
- and have a **stub op** in the process node be called per frame with real
  `FrameData` — proving the op seam works.

The framework had to make it obvious where the real ops plug in, nothing more.
`gate_pose` (2026-08-09) and `tetra_unfold` (2026-08-10) then plugged in
exactly there, and `stub` was deleted — see §5.

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
### `stub` — DELETED (2026-08-10)

The scaffolding pass shipped a `stub` op as the op-seam proof and the
op-writing template. With two real ops in the tree it was only log noise, so
it is gone from both YAMLs and from the repo; the op contract it documented
(what `create_op`, `OpContext`, `FrameData` and `draw()` give you) is now in
`vitpose_process_node.py`'s module docstring.

### `gate_pose` — SHIPPED (implementation supersedes the original sketch)

`utils/vitpose_utils.py` SECTION 2 (`FusedPlanarPoseEstimator`, ROS-free) +
`vitpose_ops/gate_pose.py`. The original sketch used the mask only as a
post-hoc validator; offline verification showed the mask should be a
**measurement source**: under bottom occlusion the surviving top keypoints
are near-collinear (keypoint-only PnP loses pitch and IPPE degenerates to
zero candidates), while the aperture's *side edges* in the mask carry
exactly the missing information. What shipped:

1. **Init**: IPPE via `solvePnPGeneric` (`solvePnPRansac` unusable — forces
   EPnP, breaks on coplanar points) with the **flip-ambiguity guard**
   (near-equal reprojection errors → prefer the candidate whose plane normal
   agrees with the prior); fallback ITERATIVE PnP seeded from the prior
   (last accepted pose within `prior_timeout`, else a canonical face-on
   guess at `prior_distance`).
2. **Fused refinement** (RAPiD-style outer/inner loop, coarse-to-fine):
   Huber LM over SE(3) with score-weighted keypoint reprojection residuals +
   dense mask-edge residuals (0.5-crossing search along projected-boundary
   normals, sharpness-weighted). Post midpoints (5, 7) contribute only their
   perpendicular-to-post component (their known sliding error mode).
   **Two-sided level test** per crossing (mask ~1 one side, ~0 the other)
   rejects contamination boundaries — without it an occluder edge near the
   true boundary captures the dense term and fusion *underperforms* kp-only.
   (Leave-one-edge-out consensus was tried and rejected: with few keypoints
   every edge is load-bearing.)
3. **Validation**: projected-aperture IoU (`min_mask_iou` 0.4; empty mask
   abstains), `max_distance`, behind-camera check. Rejected solves publish
   nothing — the object map server Kalman-filters, so dropped frames are
   cheap and wrong frames are not.
4. Outputs per config with full orientation (`rotation_quat`): `gate_link`
   (origin), `gate_entrance_link` (aperture centre (0, 0, 0.76)). NOTE:
   these names collide with the YOLO gate pipeline — intended for
   one-at-a-time operation; rename in config to run both.

Verified numbers + harness pointer: the SECTION 2 banner in
`utils/vitpose_utils.py`, and the status block above.

### `tetra_unfold` — SHIPPED 2026-08-10 (supersedes the sketch)

`vitpose_ops/tetra_unfold.py` + `utils/vitpose_utils.py` SECTION 3 +
tetra.yaml op config. Uses masks *and* keypoints; no PnP (tetra keypoints are
per-image glyph centres, not fixed 3D points).

Two deltas from the original sketch, both deliberate:

- **Chirality is a config constant** (`chirality: cw | ccw`), not something
  the op avoids assuming. We now see the object before the competition, so
  the arrangement is known. It is used *only* to lay the colours out in the
  rendered net — never in the association, which still comes purely from the
  per-frame lookup. No auto-detection, no observed-vs-config cross-check
  (considered and dropped).
- **The per-frame assignment + conflict rules became a Bayesian filter over
  the 6 letter permutations.** One-letter-per-face is then structural rather
  than a tie-break, the "third letter is inferred" case falls out of the
  marginals quantitatively, and `p_best` is a real lock criterion.

1. Per frame (`letter_face_membership`): letters below `min_score` contribute
   nothing (a letter on an averted face has no training supervision — its
   prediction is garbage, and the score gate is the only defence). Membership
   = soft mask probability sampled at the keypoint; letters outside every
   mask fall back to a nearest-mask proximity term (distance transforms
   computed lazily, only in that case). Rows normalized and floored at `eps`,
   which bounds any single frame's influence.
2. Over time (`LetterFaceFilter`): `logp ← λ·logp + gain·loglik`,
   `λ = exp(−dt/τ)`, clamped to ±`logp_clip` so a wrong lock stays reversible.
3. Publishes `tetra/letter_colors` (`std_msgs/String`, **latched**, on change
   or ≤2 Hz): `"A:red B:blue C:green p=0.97 state=LOCKED inferred=C"`.
   Precedent: `octagon/object_list`. Plus `tetra_unfold/reset` (`Empty`).
4. Publishes the unfolded net on `tetra_unfold_image/compressed` — the tetra
   net *is* a triforce (three coloured corner triangles around the white
   base), letters drawn on their faces, styled by confidence, with a footer
   table of the filter marginals. **Rendered only while subscribed.** The op
   deliberately has **no `draw()`**: the main overlay stays pure model output.

**Verification** (harness dream:~/tetra_unfold_check, real tetra_joint.pth over
the 100-image tetra_1000b val split — the checkpoint's own held-out data,
18 ms/img on the 4060 Ti, kp error 1.46 px median):

- single-frame association accuracy **100/100** — with a GT bbox the lookup is
  near-trivial, which is why the filter is stress-tested rather than credited
  by val alone;
- 30-frame sequences built from same-permutation val images: locks in
  **4 frames (0.4 s @10 Hz)**, 0 wrong locks, 100% correct;
- **stress** (real evidence corrupted per frame, fraction ρ): one letter
  landing on a neighbouring face — filtered stays **100% up to ρ=0.7** while
  raw per-frame falls to 64%; coherently-wrong frames — **100% at ρ=0.35 and
  98% at ρ=0.5**, blurred/washed-out frames only slow the lock;
- **memory sweep** (the τ/`logp_clip` decision, `memory_sweep.png`): the
  arrangement is static, so with independent frames the truth wins on
  frequency alone whenever it stays the *plurality* — under this corruption
  model each wrong world takes ρ/2, so that holds up to ρ=2/3, and indeed
  every accumulating setting converges to 100% at ρ=0.35 and ρ=0.5 while
  ρ=0.7 decays to ~6% (more memory = more confidently wrong; unfixable by
  filtering). A short window is therefore a *ceiling*, not a safeguard: the
  original τ=3 s plateaued at 98% / 84%. Forgetting still earns its keep
  because model errors are time-correlated, not independent — recovery from
  30 wrong-world frames costs 5 / 12 / 25 / 30 frames at τ = 3 / 10 / 30 / ∞.
  **τ=10 s, clip=20** takes the accuracy for 1.2 s of recovery and locks no
  slower (4 frames clean, 7 at ρ=0.35). A wrong first lock at ρ=0.5 still
  happens ~10% of the time; it is transient — 98% correct by 6 s;
- **averted letter** (one letter under the gate in every frame): the unseen
  letter's colour is recovered **100%** by the permutation constraint and
  flagged `inferred` 100% of the time;
- **recovery**: after 10 frames of a wrong world, re-locks on the truth in
  7 frames (0.7 s);
- lock criteria `0.99 / 10` were *chosen by sweep*, not guessed: vs `0.9 / 5`
  they cost one extra frame on clean data and cut wrong first-locks
  6.8% → 1.2% at ρ=0.35.
- ROS smoke: val images replayed at 10 Hz through both nodes — association
  converges, and the net topic published **0** frames while unsubscribed, 70
  after `rqt_image_view` attached.

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
    - type: gate_pose     # model_points, aperture_polygon, thresholds,
      params: {...}       # outputs — see gate.yaml and §5.
```

Future `model_points` will use the last_dance2 shape-factory spec
(`build_pose_keypoints`) — points, plus `circle`/`rect` free for other objects;
0-indexed ids like everything else (the valve YAML's 1-indexing dies with it).

`tetra.yaml` deltas: bottom camera topics, no `flip_pairs`/`flip_tta: false`,
`mask_classes: [red, green, blue]`, `keypoint_names: [A, B, C]`, the
`tetra_unfold` op, no skeleton.

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
   images (kps + masks drawn correctly on gate AND tetra), each op's outputs
   firing per frame, and the `~set_config` gate→tetra live switch.

Both shipped ops carry their own offline verification harness on dream:
`~/gate_fusion_verify` (gate_pose) and `~/tetra_unfold_check` (tetra_unfold,
including the lock-criteria sweep and the corrupted-frame stress modes).

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

- TRT engine export & backend (stub only).
- Objectness/tracker bbox provider (seam exists: Detection2DArray topic).
- SMACH integration, mission states, `robosub.launch` wiring.
- Any valve resurrection.

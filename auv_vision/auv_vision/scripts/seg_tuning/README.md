# Pipe segmentation tuning (OpenCV fallback)

Offline tools to tune the classic-OpenCV yellow-pipe segmentation that feeds
`pipe_follower_legacy`. They are the fallback for when the YOLO segmentation
model (`yolo_seg_to_mask.py`) is unavailable.

Everything here imports the same segmentation core the live node uses
(`utils/pipe_segmentation.py`), so whatever you tune is exactly what the robot
runs. See `agent_prompt.md` for the agent's iteration playbook.

## Where things run

| Component | Node / tool | Machine |
|-----------|-------------|---------|
| Mask publisher | `opencv_seg_publisher.py` (ROS node) | **robot Jetson** |
| GT drawing UI | `seg_gt_ui_node.py` (ROS node) | **operator laptop** |
| Tuning tools | `seg_eval.py`, `seg_optimize.py` (CLI) | **operator laptop** |

The Jetson node and the laptop nodes share one ROS master. On the laptop:

```bash
export ROS_MASTER_URI=http://<robot-ip>:11311
export ROS_IP=<laptop-ip>
```

Use `~use_compressed:=true` on the laptop UI node to pull compressed camera
frames over the network.

## End-to-end flow

1. **Bring up the fallback on the robot** (Jetson):
   ```bash
   roslaunch auv_vision opencv_seg_publisher.launch
   ```
   Publishes `/taluy/bottle_mask`. Live-tune with `rqt_reconfigure`.

2. **Open the GT UI on the laptop**:
   ```bash
   roslaunch auv_vision seg_gt_ui.launch use_compressed:=true
   ```
   Open <http://localhost:8088/>, click **Grab live frame** (or **Upload**), paint
   the pipe, **Save to dataset**. Repeat for several frames (near + far + tricky):
   each save appends `/tmp/seg_session/sample_NNNN/{image.png,gt.png}`. **Reset
   dataset** clears them.

3. **Optimize** against the whole dataset (averages IoU over all samples, so it
   does not overfit to one near/far frame):
   ```bash
   python3 seg_optimize.py --session /tmp/seg_session --out best_params.yaml --eval
   ```
   `--session` auto-detects a single `image.png`+`gt.png` pair or a directory of
   `sample_*/` subdirs. You can also pass explicit pairs:
   ```bash
   python3 seg_optimize.py near.png far.png --gt near_gt.png --gt far_gt.png \
           --out best_params.yaml
   ```

4. **Inspect / diagnose** the result:
   ```bash
   python3 seg_eval.py /tmp/seg_session/sample_0001/image.png \
           --params best_params.yaml --gt /tmp/seg_session/sample_0001/gt.png --diag
   ```
   Writes `*_mask.png`, `*_overlay.png` (original | mask | overlay) and, with
   `--diag`, `*_diag.png` (original | GT | pred | error; green=correct,
   red=false-positive, blue=false-negative).

5. **Push to the live node** over the ROS network:
   ```bash
   rosrun dynamic_reconfigure dynparam load /taluy/opencv_seg_publisher best_params.yaml
   ```
   The values update live in `rqt_reconfigure`; `pipe_follower_legacy` consumes
   the improved mask immediately. View it with
   `rqt_image_view /taluy/bottle_mask`.

## Param schema

`best_params.yaml` is a flat dict using the dynamic-reconfigure schema (so it is
directly `dynparam load`-able). Field meanings live in
`utils/pipe_segmentation.py` (`SegParams`) and `cfg/OpenCVSeg.cfg`.
`combine_mode` is an int: `0=hsv_only 1=lab_only 2=and 3=or`.

## Notes

- The GT UI uses only the Python standard library (`http.server`) — no Flask.
  If you prefer Flask, the `frame_provider`/`/save`/`/list`/`/clear` contract in
  `gt_web/server.py` is small enough to reimplement.
- These tools run in place (`python3 seg_tuning/<tool>.py`); they are not
  installed by catkin.

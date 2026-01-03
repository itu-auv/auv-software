# Slalom Perception Pipeline - Project Log

## Goal

Implement a depth-based pipe segmentation system for Robosub 2025 Slalom task using Depth Anything V3 + RGB sensor fusion.

## Plan

1. Add custom ROS messages (`ObjectDetection`, `ObjectDetectionArray`) to `auv_msgs`.
2. Convert `auv_vision` into a proper Python package (add `setup.py`, `src/` structure).
3. Implement `segment_slalom_pipes()` in a reusable module.
4. Integrate into existing `depth_anything_client.py` via a simple import.

## What Worked

- Otsu thresholding on normalized depth effectively separates foreground objects.
- Vertical morphological kernel (5x15) reinforces pipe shapes.
- HSV-based color classification is robust to underwater blue tint.
- Dataclass-based `PipeDetection` keeps the API clean.
- Synthetic unit test confirms correct detection (bbox, color, depth).
- **Docker Build**: `catkin build` succeeded inside `dockauv` container.
- **Integration**: Python package `auv_vision` import verification passed in Docker.

## What Didn't Work

- `catkin build` on local host (outside Docker) fails due to system `librt.so` issue. Use `dockauv` for building and running.

## Next Steps

- [ ] Run `dockauv rosrun auv_vision depth_anything_client.py _enable_slalom:=true` to test with camera stream.
- [ ] Verify message publishing on `/perception/slalom_pipes`.
- [ ] Tune thresholds (`MIN_CONTOUR_AREA`, `MIN_ASPECT_RATIO`) for competition pool conditions.
- [ ] Add temporal filtering to reduce flickering detections.

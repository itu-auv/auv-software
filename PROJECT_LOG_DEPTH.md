# Depth-Anything-3 + ZMQ Integration Project Log

## Goal

ZMQ-based depth inference: DA3 server on GPU, ROS client publishes depth + colorized.

## What Worked

- Migrated to `itu-auv/Depth-Anything-3` fork (branch: `itu-auv-main`)
- Preserved `zmq_server.py` and `zmq_client.py`
- Simplified `depth_anything_client.py`:
  - Rate-based loop (no complex batching)
  - Intrinsics fetch (non-blocking)
  - Topics: `~/depth` (32FC1), `~/colorized` (bgr8 INFERNO)
- Verified end-to-end functionality via Docker client + Host server
- Created `auv_vision/launch/depth_anything.launch`

## Next

- Integrate into full AUV startup sequence

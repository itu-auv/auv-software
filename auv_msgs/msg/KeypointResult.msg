# Keypoint detection result for a single image frame.

std_msgs/Header header
Keypoint[] keypoints

# VITTrack bounding box [x, y, w, h] in image pixels, carried through purely so
# the pose node can draw it on the debug image. Empty when the producer has no
# tracked box this frame (UNINIT / RECOVERING / tracker lost).
float64[] bbox

# Producer tracker state machine: "UNINIT" / "LOCKED" / "RECOVERING".
# Display-only; lets the pose-node debug image show what the producer is doing.
string state

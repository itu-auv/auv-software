# One ViTPose inference result: keypoints + segmentation for a single object
# instance. Produced by auv_vision/vitpose_detection_node, consumed by
# vitpose_process_node. Full contract: auv_vision/VITPOSE_PLAN.md.

std_msgs/Header header

# Config object name this result belongs to, e.g. "gate", "tetra".
string object

# [x, y, w, h] in source-image pixels: the bbox fed to the model.
# Empty when the whole frame was used.
float64[] bbox

# All K keypoints, always present, 0-indexed ids, source-image pixels,
# raw heatmap-peak confidences. Consumers apply their own confidence gates.
Keypoint[] keypoints

# Index-aligned with keypoint ids (keypoint_names[k.id] names keypoint k).
string[] keypoint_names

# Mask class names in model channel order (= profile declaration order).
string[] mask_classes

# One mono8 probability map per mask class (0..255 = probability 0..1),
# already inverse-warped to source-image resolution. Index-aligned with
# mask_classes. Threshold with mask_threshold for binary masks; never
# binarize before any further warping.
sensor_msgs/Image[] masks

# Calibrated binarization threshold for the masks (from the checkpoint).
float32 mask_threshold

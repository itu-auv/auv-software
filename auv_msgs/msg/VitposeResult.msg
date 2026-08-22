# One ViTPose inference result for a single object instance.
# Producer: auv_vision/vitpose_detection_node; consumer: vitpose_process_node.

std_msgs/Header header
string object                 # config object name, e.g. "gate", "tetra"
float64[] bbox                # [x, y, w, h] px fed to the model; empty = full frame
auv_msgs/Keypoint[] keypoints # all K, 0-indexed ids, source px, raw confidences
string[] keypoint_names       # keypoint_names[k.id] names keypoint k
string[] mask_classes         # model channel order
sensor_msgs/Image[] masks     # mono8 probability maps (0..255 = 0..1), source
                              # resolution, aligned with mask_classes; threshold
                              # with mask_threshold AFTER any further warping
float32 mask_threshold        # calibrated binarization threshold (checkpoint)

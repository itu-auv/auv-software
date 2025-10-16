# RL Agent için State Vector
Header header

# Detection features (max 10 object için)
# Her obje: [normalized_x, normalized_y, normalized_width, normalized_height, distance, angle_x, angle_y, confidence, object_id]
float32[] detection_features  # boyut: MAX_OBJECTS * 9

# Vehicle state
float32 depth
float32 altitude
float32 roll
float32 pitch
float32 yaw
float32 velocity_x
float32 velocity_y
float32 velocity_z
float32 angular_vel_z

# Mission info
string target_object  # "gate", "buoy", "torpedo" etc.
float32 mission_progress  # 0.0 to 1.0

# Metadata
uint8 num_detections

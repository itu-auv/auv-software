# Represents a collection of detected pipes at a specific time and in a specific frame

# Standard ROS header (timestamp, frame_id)
std_msgs/Header header

# Array of detected pipes
auv_msgs/Pipe[] detected_pipes

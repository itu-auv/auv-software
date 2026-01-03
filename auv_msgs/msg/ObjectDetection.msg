# Single detected object (pipe, gate, etc.)

string label
string color
float32 confidence

# Bounding box [x, y, w, h] in pixels
int32[] bbox

# Median depth value within the mask
float32 depth

# Centroid in normalized image coords, range [-1, 1]
geometry_msgs/Point centroid

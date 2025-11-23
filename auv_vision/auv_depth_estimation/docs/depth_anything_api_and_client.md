# Depth Anything 3 ROS Implementation - Architecture & API Documentation

## 📐 System Architecture

Our depth estimation system uses a **client-server architecture** with ZeroMQ for high-performance, low-latency communication between ROS and the GPU-accelerated inference backend.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          ROS ECOSYSTEM                              │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  ROS Node: depth_anything_client.py                        │   │
│  │  (Python 3.8 - System Python for ROS Noetic)              │   │
│  │                                                            │   │
│  │  ┌──────────────┐     ┌──────────────┐                    │   │
│  │  │   Subscribe  │────>│ Image Buffer │                    │   │
│  │  │  /camera/    │     │   + TF2      │                    │   │
│  │  │  image_raw   │     │  Transforms  │                    │   │
│  │  └──────────────┘     └──────┬───────┘                    │   │
│  │                              │                             │   │
│  │                              ▼                             │   │
│  │                     ┌─────────────────┐                    │   │
│  │                     │ Batch Selection │                    │   │
│  │                     │ (Dynamic)       │                    │   │
│  │                     └────────┬────────┘                    │   │
│  │                              │                             │   │
│  │                              ▼                             │   │
│  │                  ┌───────────────────────┐                 │   │
│  │                  │   ZeroMQ REQ Client   │                 │   │
│  │                  │  tcp://host:5555      │                 │   │
│  │                  └───────────┬───────────┘                 │   │
│  │                              │ Binary NumPy                │   │
│  └──────────────────────────────┼─────────────────────────────┘   │
│                                 │                                 │
│                  ┌──────────────▼──────────────┐                  │
│                  │  Publish:                   │                  │
│                  │  • /depth_anything/depth    │                  │
│                  │  • /depth_anything/points   │                  │
│                  └─────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ ZeroMQ over TCP
                                 │ (localhost or network)
                                 │
┌─────────────────────────────────▼───────────────────────────────────┐
│                     INFERENCE BACKEND                               │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  ZeroMQ Server: zmq_server.py                              │   │
│  │  (Python 3.12 - Virtual Environment)                       │   │
│  │                                                            │   │
│  │  ┌───────────────────┐                                     │   │
│  │  │  ZeroMQ REP Server│                                     │   │
│  │  │  tcp://*:5555     │                                     │   │
│  │  └─────────┬─────────┘                                     │   │
│  │            │                                                │   │
│  │            ▼                                                │   │
│  │  ┌─────────────────────────────┐                           │   │
│  │  │  Request Router              │                           │   │
│  │  │  • ping                      │                           │   │
│  │  │  • inference                 │                           │   │
│  │  │  • inference_batch           │                           │   │
│  │  └────────────┬─────────────────┘                           │   │
│  │               │                                             │   │
│  │               ▼                                             │   │
│  │  ┌──────────────────────────────┐                          │   │
│  │  │  Depth Anything 3 Model      │                          │   │
│  │  │  • Loaded in GPU memory      │                          │   │
│  │  │  • da3-small / base /        │                          │   │
│  │  │    large / giant             │                          │   │
│  │  │  • Persistent across requests│                          │   │
│  │  └────────────┬─────────────────┘                          │   │
│  │               │                                             │   │
│  │               ▼                                             │   │
│  │  ┌──────────────────────────────┐                          │   │
│  │  │  Returns:                     │                          │   │
│  │  │  • depth: (H, W) float32     │                          │   │
│  │  │  • intrinsics: (3, 3) float32│                          │   │
│  │  └──────────────────────────────┘                          │   │
│  └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔄 Data Flow

### 1. Image Acquisition (ROS Client)
```python
# Subscribe to camera topic
/camera/image_raw (sensor_msgs/Image)
    ↓
# Convert to OpenCV BGR format
cv_bridge.imgmsg_to_cv2(msg, "bgr8")
    ↓
# Fetch camera intrinsics (once at startup)
CameraCalibrationFetcher → K (3×3 matrix)
    ↓
# Lookup camera extrinsics via TF2 (if use_external_odom=true)
TF2: odom → camera_frame → 4×4 transformation matrix
    ↓
# Buffer: (timestamp, image, header, extrinsics)
collections.deque(maxlen=100)
```

### 2. Dynamic Batching
```python
# Select frames from buffer
Context Window (0.5s) → Valid frames
    ↓
# Uniform sampling
np.linspace(0, N-1, batch_size) → Indices
    ↓
# Prepare batch
images: List[ndarray(H, W, 3)]       # RGB
intrinsics: List[ndarray(3, 3)]      # Camera K matrices
extrinsics: List[ndarray(4, 4)]      # World-to-camera transforms
```

### 3. ZeroMQ Communication
```python
# Client (REQ socket)
request = {
    'command': 'inference_batch',
    'images': [rgb1, rgb2, ...],        # List of RGB arrays
    'intrinsics': [K1, K2, ...],        # List of 3×3 matrices (optional)
    'extrinsics': [T1, T2, ...],        # List of 4×4 matrices (optional)
    'process_res': 504
}
socket.send_pyobj(request, protocol=2)  # Pickle protocol 2
    ↓
# Server (REP socket)
receives → unpickles → processes
    ↓
response = {
    'status': 'success',
    'depth': [depth1, depth2, ...],     # List of (H, W) float32 arrays
    'intrinsics': [K1, K2, ...]         # List of refined 3×3 matrices
}
socket.send_pyobj(response)
```

### 4. Inference (Backend Server)
```python
# Convert to model format
images → torch.Tensor (GPU)
    ↓
# Forward pass
with torch.no_grad():
    prediction = model.inference(
        images,
        intrinsics=intrinsics,      # Optional: enables pose-aware
        extrinsics=extrinsics,      # Optional: multi-view consistency
        process_res=504,
        export_format="mini_npz"
    )
    ↓
# Extract results
depth: List[ndarray(H, W)]          # Metric depth in meters
intrinsics: List[ndarray(3, 3)]     # Refined camera intrinsics
```

### 5. Point Cloud Generation (ROS Client)
```python
# Unproject depth to 3D
Z = depth
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
    ↓
# Apply color from RGB image
RGB → uint8 → pack to XYZRGB format
    ↓
# Publish
/depth_anything/depth (sensor_msgs/Image, 32FC1)
/depth_anything/points (sensor_msgs/PointCloud2)
```

## 🔌 API Reference

### Server API (zmq_server.py)

#### Command: `ping`
Health check for server availability.

**Request:**
```python
{
    'command': 'ping'
}
```

**Response:**
```python
{
    'status': 'success',
    'message': 'pong'
}
```

---

#### Command: `inference`
Single image depth estimation.

**Request:**
```python
{
    'command': 'inference',
    'image': ndarray(H, W, 3),      # RGB uint8
    'process_res': 504               # Optional, default: 504
}
```

**Response:**
```python
{
    'status': 'success',
    'depth': ndarray(H, W),          # float32, metric depth in meters
    'intrinsics': ndarray(3, 3)      # float32, camera intrinsics matrix
}
```

---

#### Command: `inference_batch`
Batch depth estimation with optional pose conditioning.

**Request:**
```python
{
    'command': 'inference_batch',
    'images': [                              # List of RGB images
        ndarray(H1, W1, 3),
        ndarray(H2, W2, 3),
        ...
    ],
    'intrinsics': [                          # Optional: camera intrinsics
        ndarray(3, 3),                       # K matrix for image 1
        ndarray(3, 3),                       # K matrix for image 2
        ...
    ],
    'extrinsics': [                          # Optional: camera poses
        ndarray(4, 4),                       # World→Cam transform for image 1
        ndarray(4, 4),                       # World→Cam transform for image 2
        ...
    ],
    'process_res': 504                       # Optional, default: 504
}
```

**Response:**
```python
{
    'status': 'success',
    'depth': [                               # List of depth maps
        ndarray(H1, W1),                     # float32
        ndarray(H2, W2),
        ...
    ],
    'intrinsics': [                          # List of refined intrinsics
        ndarray(3, 3),                       # float32
        ndarray(3, 3),
        ...
    ]
}
```

**Notes:**
- Both `intrinsics` and `extrinsics` must be provided together to enable pose-conditioned mode
- If either is `None`, the model operates in single-view mode
- Batch size is limited by GPU memory (typically 2-10 frames for da3-large)

---

### Client API (ROS Parameters)

#### Connection Parameters
- `~zmq_host` (string, default: `"localhost"`)
  Hostname or IP of the ZeroMQ server

- `~zmq_port` (int, default: `5555`)
  Port number of the ZeroMQ server

#### Topic Parameters
- `~input_topic` (string, default: `"/camera/image_raw"`)
  Input camera topic (sensor_msgs/Image)

- `~output_topic` (string, default: `"/depth_anything/points"`)
  Output point cloud topic (sensor_msgs/PointCloud2)

- `~depth_topic` (string, default: `"/depth_anything/depth"`)
  Output depth map topic (sensor_msgs/Image, 32FC1)

#### Processing Parameters
- `~process_res` (int, default: `504`)
  Processing resolution (higher = better quality, slower)

- `~max_batch_size` (int, default: `2`)
  Maximum number of frames to batch together

- `~context_window` (float, default: `0.5`)
  Time window in seconds to look back for context frames

#### Camera Parameters
- `~camera_namespace` (string, default: `"/taluy/cameras/cam_front"`)
  Namespace for camera_info topic

- `~camera_frame` (string, default: `"taluy/base_link/front_camera_optical_link"`)
  TF frame of the camera (optical frame)

- `~odom_frame` (string, default: `"odom"`)
  TF frame for odometry reference

#### Feature Flags
- `~use_external_odom` (bool, default: `false`)
  Enable pose-conditioned depth estimation using TF2 transforms

---

## 🧮 Data Structures

### Camera Intrinsics Matrix (K)
```
     ┌            ┐
     │ fx  0  cx  │
K =  │ 0  fy  cy  │
     │ 0   0   1  │
     └            ┘
```
- `fx`, `fy`: Focal lengths in pixels
- `cx`, `cy`: Principal point (image center)

### Camera Extrinsics Matrix (T)
```
     ┌                    ┐
     │ r11  r12  r13  tx  │
T =  │ r21  r22  r23  ty  │  (World → Camera)
     │ r31  r32  r33  tz  │
     │  0    0    0    1  │
     └                    ┘
```
- `R` (3×3): Rotation matrix (SO(3))
- `t` (3×1): Translation vector in meters
- Converts world coordinates to camera coordinates: `P_cam = T @ P_world`

---

## 🚀 Performance Characteristics

### Latency Breakdown
```
Total latency: ~15-30ms per frame

├─ Image acquisition:       1-2ms   (ROS subscriber)
├─ Batching & prep:        0.5-1ms  (NumPy operations)
├─ ZeroMQ serialization:   1-2ms    (pickle)
├─ Network transfer:       <1ms     (localhost TCP)
├─ GPU inference:          8-15ms   (da3-large @ 504px)
├─ ZeroMQ deserialization: 1-2ms    (unpickle)
└─ Point cloud generation: 2-5ms    (unprojection)
```

### Throughput
- **Single image mode**: 30-60 FPS (depending on resolution)
- **Batch mode**: 15-30 FPS (amortized per image)
- **GPU memory**: ~4GB for da3-large

### Comparison with HTTP Backend
| Metric | ZeroMQ | HTTP |
|--------|--------|------|
| Latency | 15-30ms | 50-100ms |
| Serialization | Binary (pickle) | JSON + Base64 |
| Overhead | Minimal | HTTP headers + encoding |
| Throughput | 15-30 FPS | 10-20 FPS |
| Connection | Persistent socket | Request-response |
| Complexity | Custom server | Built-in API server |

---

## 🔧 Implementation Details

### ZeroMQ Socket Configuration
```python
# Client (REQ socket)
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{host}:{port}")
socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30s receive timeout
socket.setsockopt(zmq.SNDTIMEO, 5000)   # 5s send timeout
socket.setsockopt(zmq.LINGER, 0)        # Don't wait on close
```

```python
# Server (REP socket)
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")
```

### TF2 Transform Lookup
```python
# Lookup camera pose at image timestamp
transform = tf_buffer.lookup_transform(
    target_frame='odom',
    source_frame='camera_optical_link',
    time=image_timestamp,
    timeout=Duration(0.1)
)

# Convert to 4×4 matrix
T = quaternion_matrix([qx, qy, qz, qw])
T[0:3, 3] = [tx, ty, tz]
```

### Dynamic Batch Selection
```python
# Filter frames within context window
valid_frames = [
    f for f in buffer
    if (current_time - f.timestamp) <= context_window
]

# Uniform sampling with stride
indices = np.linspace(0, len(valid_frames)-1, batch_size, dtype=int)
batch = [valid_frames[i] for i in indices]
```

---

## 🎯 Use Cases

### 1. Real-time Obstacle Avoidance (Single Image)
```bash
roslaunch auv_depth_estimation depth_anything.launch \
    max_batch_size:=1 \
    context_window:=0.1
```
- Minimal latency
- No multi-view processing
- Best for reactive control

### 2. Mapping & SLAM (Multi-view with Poses)
```bash
roslaunch auv_depth_estimation depth_anything.launch \
    use_external_odom:=true \
    max_batch_size:=5 \
    context_window:=0.5
```
- Leverages camera motion
- Improved depth consistency
- Better for reconstruction

### 3. High-throughput Processing (Batch Mode)
```bash
roslaunch auv_depth_estimation depth_anything.launch \
    max_batch_size:=10 \
    context_window:=1.0
```
- Amortizes GPU overhead
- Higher throughput
- Acceptable latency increase

---

## 🐛 Debugging

### Check Server Status
```bash
python3 -c "
import zmq
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect('tcp://localhost:5555')
sock.send_pyobj({'command': 'ping'})
print(sock.recv_pyobj())
"
```

### Monitor Topics
```bash
# Check publishing rates
rostopic hz /depth_anything/depth
rostopic hz /depth_anything/points

# Inspect messages
rostopic echo /depth_anything/depth
```

### Verify TF Tree
```bash
# View TF tree
rosrun tf view_frames

# Check specific transform
rosrun tf tf_echo odom camera_optical_link
```

### Server Logs
```bash
# Enable debug logging in zmq_server.py
logging.basicConfig(level=logging.DEBUG)
```

---

## 📊 Monitoring

### Server-side Metrics
- VRAM usage (logged every request)
- Request count (logged every 100 requests)
- Processing time per batch

### Client-side Metrics
- Frame buffer status
- Batch selection details
- End-to-end latency
- TF lookup failures

---

## 🔒 Error Handling

### Connection Failures
- **Timeout**: Server not responding → Check if server is running
- **Connection refused**: Wrong host/port → Verify ZeroMQ configuration
- **Socket closed**: Server crashed → Restart server

### TF Lookup Failures
- **LookupException**: Transform not available → Check TF tree with `view_frames`
- **ExtrapolationException**: Timestamp too old/new → Increase TF buffer size
- **ConnectivityException**: Broken TF chain → Verify frame names

### Inference Failures
- **CUDA OOM**: Batch too large → Reduce `max_batch_size` or `process_res`
- **Invalid shape**: Image size mismatch → Check camera resolution
- **Pickle error**: Protocol mismatch → Use `protocol=2` for compatibility

---


# Depth Anything 3 ROS: Architecture & API

This package bridges ROS (Noetic) with **Depth Anything 3** using a high-performance **ZeroMQ** client-server architecture.

**The Gist:** We moved the heavy inference out of the ROS node. The ROS node acts as a lightweight client that ships images (and optionally poses) to a dedicated Python server keeping the model "hot" in GPU memory.

## 🏗 System Architecture

We ditched HTTP for **ZeroMQ (TCP)**. Why?

1.  **Speed:** \~15ms latency (vs. \~100ms with HTTP).
2.  **Efficiency:** Binary serialization (Pickle) means no JSON overhead.
3.  **State:** The model stays loaded in VRAM; no reload penalty per request.

### The Pipeline

1.  **Capture:** `depth_anything_client.py` grabs `/camera/image_raw`.
2.  **Context (Optional):** If `use_external_odom` is on, we fetch the camera intrinsics ($K$) and look up the extrinsics ($T$) via TF2 (`odom` → `camera_optical_frame`).
3.  **Batching:** Frames are buffered and batched dynamically based on a time window (default 0.5s).
4.  **Transport:** The batch is pickled and shot over ZMQ to the server.
5.  **Inference:** The server runs the model (optionally using the pose data for better accuracy) and returns depth maps.
6.  **Publish:** The client unprojects the depth to 3D and publishes `/depth_anything/points` (PointCloud2).

-----

## 🔌 API Reference

### 1\. The Server (`zmq_server.py`)

The server listens on `tcp://*:5555`. It speaks **Pickle protocol 2**.

**Primary Command: `inference_batch`**
Send this dict to the server:

```python
{
    'command': 'inference_batch',
    'images': [np_array_rgb, ...],    # List of images
    'intrinsics': [K_matrix, ...],    # Optional: List of 3x3 matrices
    'extrinsics': [T_matrix, ...],    # Optional: List of 4x4 matrices
    'process_res': 504                # Resolution (default 504)
}
```

*Note: To use the "pose-conditioned" features of Depth Anything 3, you MUST provide both intrinsics and extrinsics.*

### 2\. The Client (ROS Parameters)

Configure these in your launch file:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `zmq_host` | `localhost` | Where the inference server lives. |
| `input_topic` | `/camera/image_raw` | Your camera source. |
| `process_res` | `504` | Higher = better quality, slower. |
| `max_batch_size` | `2` | Max frames to send at once. Watch your VRAM. |
| `use_external_odom`| `false` | **Critical:** Set `true` to enable pose-aware depth. |



#!/usr/bin/env python3
"""
Auto-Labeler for Valve Dataset (Images + YOLO Segmentation Labels).

Spawns a free-floating camera in Gazebo, teleports it through a dense
hemisphere grid of poses around the valve, captures images AND generates
automatic YOLO segmentation labels via 3D→2D projection.

The valve's 3D circle (radius=120mm, front face at X=+20mm in mesh coords)
is projected onto each camera image using known geometry and camera intrinsics.
This produces pixel-perfect segmentation polygons with zero manual labeling.

Output structure (ready for YOLOv8 training):
    <output_dir>/
        train/
            images/   *.jpg
            labels/   *.txt  (YOLO seg format: class x1 y1 x2 y2 ... xN yN)
        valid/
            images/   *.jpg
            labels/   *.txt
        data.yaml

Usage:
    rosrun auv_vision sim_auto_labeler.py _model_name:=tac_valve_alone
"""

import os
import random
import glob
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, SetModelStateRequest
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_matrix, euler_matrix, quaternion_matrix

# ---------------------------------------------------------------------------
# Valve geometry (from DAE analysis, in METERS after 0.001 scale)
# ---------------------------------------------------------------------------
VALVE_FACE_CENTER_LOCAL = np.array([0.020, 0.0, 0.0])   # front face center in model coords
VALVE_RADIUS_M = 0.120                                    # outer ring radius
CIRCLE_POINTS = 72                                        # polygon resolution (every 5°)

# tac_valve specific offset (the valve link within the tac_valve station)
VALVE_RIGHT = {"pos": [0.58, 0.555, 1.4205], "rpy": [0.0, np.pi, 0.0]}

# ---------------------------------------------------------------------------
# Camera intrinsics (must match _CAMERA_SDF below)
# ---------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480
HFOV = 1.229  # radians
FX = IMG_W / (2.0 * np.tan(HFOV / 2.0))  # ~497.2
FY = FX  # square pixels
CX, CY = IMG_W / 2.0, IMG_H / 2.0

# ---------------------------------------------------------------------------
# Pose grid — Dense hemisphere on the face-normal side of the valve.
# ---------------------------------------------------------------------------
RADII_M     = [0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0]  # 7 range steps
POLAR_DEG   = [0, 15, 30, 45, 60]                     # polar angles
AZIMUTH_DEG = list(range(0, 360, 30))                  # 12 azimuths

VALID_SPLIT_RATIO = 0.15  # 15% validation

# ---------------------------------------------------------------------------
# Standalone camera SDF
# ---------------------------------------------------------------------------
_CAMERA_SDF = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='labeler_camera'>
    <static>true</static>
    <link name='link'>
      <visual name='body'>
        <pose>-0.08 0 0 0 0 0</pose>
        <geometry><box><size>0.12 0.07 0.07</size></box></geometry>
        <material>
          <ambient>1 1 0 1</ambient>
          <diffuse>1 1 0 1</diffuse>
        </material>
      </visual>
      <sensor name='camera' type='camera'>
        <update_rate>30</update_rate>
        <camera>
          <horizontal_fov>1.229</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>B8G8R8</format>
          </image>
          <clip>
            <near>0.02</near>
            <far>300</far>
          </clip>
        </camera>
        <plugin name='camera_controller' filename='libgazebo_ros_camera.so'>
          <alwaysOn>true</alwaysOn>
          <updateRate>30</updateRate>
          <cameraName>labeler_camera</cameraName>
          <imageTopicName>image_raw</imageTopicName>
          <cameraInfoTopicName>camera_info</cameraInfoTopicName>
          <frameName>link</frameName>
          <distortionK1>0</distortionK1>
          <distortionK2>0</distortionK2>
          <distortionK3>0</distortionK3>
          <distortionT1>0</distortionT1>
          <distortionT2>0</distortionT2>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>"""


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def build_camera_rotation(p_cam, look_at, roll_rad=0.0):
    """Return 3×3 rotation matrix R: columns = [forward, right, up] in world."""
    fwd = _normalize(look_at - p_cam)
    ref = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = _normalize(np.cross(ref, fwd))
    up = np.cross(fwd, right)

    R = np.column_stack([fwd, right, up])

    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    return R @ Rx


def rotation_to_quaternion(R):
    """Convert 3×3 rotation matrix to (x, y, z, w) quaternion."""
    M = np.eye(4)
    M[:3, :3] = R
    return quaternion_from_matrix(M)


def generate_circle_points_3d(center, normal, radius, n_points):
    """
    Generate n_points on a circle in 3D space.
    
    Args:
        center: (3,) center of circle
        normal: (3,) normal vector of the plane
        radius: radius of circle
        n_points: number of points
    
    Returns: (n_points, 3) array of 3D points
    """
    normal = _normalize(normal)

    # Create two perpendicular vectors in the circle's plane
    if abs(normal[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])

    u = _normalize(np.cross(normal, ref))
    v = np.cross(normal, u)

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.zeros((n_points, 3))
    for i, a in enumerate(angles):
        points[i] = center + radius * (np.cos(a) * u + np.sin(a) * v)

    return points


def project_3d_to_2d(points_world, R_cam, t_cam):
    """
    Project 3D world points to 2D pixel coordinates.
    
    Gazebo camera frame: X=forward, Y=left, Z=up
    OpenCV image: u=right, v=down
    
    Args:
        points_world: (N, 3) world coordinates
        R_cam: (3, 3) camera rotation matrix (columns = fwd, right, up in world)
        t_cam: (3,) camera position in world
    
    Returns:
        pixels: (N, 2) pixel coordinates [u, v], or None if any point is behind camera
    """
    # Transform world points to camera frame
    # P_cam = R_cam^T @ (P_world - t_cam)
    points_cam = (points_world - t_cam) @ R_cam  # equivalent to R^T @ (p - t) for each row

    # Check all points are in front of camera (X_cam > 0)
    if np.any(points_cam[:, 0] <= 0.01):
        return None

    # Gazebo camera → image projection
    # u = fx * (-Y_cam / X_cam) + cx
    # v = fy * (-Z_cam / X_cam) + cy
    u = FX * (-points_cam[:, 1] / points_cam[:, 0]) + CX
    v = FY * (-points_cam[:, 2] / points_cam[:, 0]) + CY

    return np.column_stack([u, v])


def pixels_to_yolo_seg(pixels, img_w, img_h, class_id=0, min_area_px=100):
    """
    Convert pixel polygon to YOLO segmentation label format.
    
    Returns: label string or None if polygon is too small / outside image.
    """
    # Clip to image boundaries
    pixels_clipped = pixels.copy()
    pixels_clipped[:, 0] = np.clip(pixels_clipped[:, 0], 0, img_w - 1)
    pixels_clipped[:, 1] = np.clip(pixels_clipped[:, 1], 0, img_h - 1)

    # Check if polygon has enough area
    contour = pixels_clipped.astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(contour)
    if area < min_area_px:
        return None

    # Normalize to [0, 1]
    norm_x = pixels_clipped[:, 0] / img_w
    norm_y = pixels_clipped[:, 1] / img_h

    # Format: class_id x1 y1 x2 y2 ... xN yN
    coords = []
    for x, y in zip(norm_x, norm_y):
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")

    return f"{class_id} " + " ".join(coords)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rospy.init_node("sim_auto_labeler")

    output_dir = rospy.get_param("~output_dir", "/auv_ws/src/auv_software/valve_training/sim_dataset")

    # Create train/valid directory structure
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Clean old files
    for split in ["train", "valid"]:
        for ext in ["*.jpg", "*.txt"]:
            for f in glob.glob(os.path.join(output_dir, split, "images", ext)):
                os.remove(f)
            for f in glob.glob(os.path.join(output_dir, split, "labels", ext)):
                os.remove(f)
    rospy.loginfo("Cleared existing dataset.")

    bridge = CvBridge()

    # Wait for Gazebo services
    rospy.loginfo("Waiting for Gazebo services...")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/set_model_state")

    spawn_proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    # Spawn camera
    rospy.loginfo("Spawning labeler_camera...")
    try:
        delete_proxy("labeler_camera")
        rospy.sleep(0.5)
    except:
        pass

    spawn_proxy(
        model_name="labeler_camera",
        model_xml=_CAMERA_SDF,
        robot_namespace="",
        initial_pose=Pose(),
        reference_frame="world",
    )
    rospy.sleep(1.0)

    # Get valve pose from Gazebo
    target_model_name = rospy.get_param("~model_name", "tac_valve_alone")
    rospy.loginfo(f"Reading pose from model_states for: {target_model_name}...")

    valve_pose = None

    def _ms_cb(msg):
        nonlocal valve_pose
        for i, name in enumerate(msg.name):
            if name == target_model_name:
                valve_pose = msg.pose[i]

    sub = rospy.Subscriber("/gazebo/model_states", ModelStates, _ms_cb)
    deadline = rospy.Time.now() + rospy.Duration(5.0)

    while valve_pose is None and rospy.Time.now() < deadline:
        rospy.sleep(0.05)
    sub.unregister()

    if valve_pose is None:
        rospy.logerr(f"Could not get {target_model_name} pose — is the simulation running?")
        return

    p = valve_pose.position
    q = valve_pose.orientation
    R_model = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    t_model = np.array([p.x, p.y, p.z])
    rospy.loginfo(f"Valve pose: pos=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})")

    # Compute valve face center and normal in WORLD coordinates
    if target_model_name == "tac_valve":
        roll, pitch, yaw = VALVE_RIGHT["rpy"]
        R_link = euler_matrix(roll, pitch, yaw, axes="sxyz")[:3, :3]
        t_link = np.array(VALVE_RIGHT["pos"])
    else:
        R_link = np.eye(3)
        t_link = np.zeros(3)

    # Face center in model frame, then in world
    face_center_model = R_link @ VALVE_FACE_CENTER_LOCAL + t_link
    face_center_world = R_model @ face_center_model + t_model

    # Face normal in model frame: +X direction
    face_normal_model = R_link @ np.array([1.0, 0.0, 0.0])
    face_normal_world = R_model @ face_normal_model

    # Generate the 3D circle points on the valve's outer ring (world coords)
    # The circle lies in the plane perpendicular to face_normal, centered at face_center
    valve_circle_world = generate_circle_points_3d(
        face_center_world, face_normal_world, VALVE_RADIUS_M, CIRCLE_POINTS
    )

    rospy.loginfo(f"Valve face center (world): ({face_center_world[0]:.3f}, "
                  f"{face_center_world[1]:.3f}, {face_center_world[2]:.3f})")
    rospy.loginfo(f"Valve face normal (world): ({face_normal_world[0]:.3f}, "
                  f"{face_normal_world[1]:.3f}, {face_normal_world[2]:.3f})")

    # Camera approach direction: same as face normal (camera stands in front of the face)
    cam_approach_dir = _normalize(face_normal_world)

    # Build up/right vectors for hemisphere grid
    ref = np.array([0.0, 0.0, 1.0]) if abs(cam_approach_dir[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    grid_right = _normalize(np.cross(ref, cam_approach_dir))
    grid_up = np.cross(cam_approach_dir, grid_right)

    # Generate all poses (only "aimed" orientation — camera always looks at face center)
    poses = [
        (r, polar, az)
        for r in RADII_M
        for polar in POLAR_DEG
        for az in ([0] if polar == 0 else AZIMUTH_DEG)
    ]
    total = len(poses)
    rospy.loginfo(f"Grid: {total} images to capture...")

    # Pre-determine train/valid split
    indices = list(range(total))
    random.shuffle(indices)
    n_valid = max(1, int(total * VALID_SPLIT_RATIO))
    valid_set = set(indices[:n_valid])

    image_count = 0
    label_count = 0

    for idx, (r, polar_deg, az_deg) in enumerate(poses):
        if rospy.is_shutdown():
            break

        theta = np.radians(polar_deg)
        phi = np.radians(az_deg)

        # Camera position on hemisphere around face center
        p_cam = face_center_world + r * (
            cam_approach_dir * np.cos(theta) +
            grid_right * np.sin(theta) * np.cos(phi) +
            grid_up * np.sin(theta) * np.sin(phi)
        )

        # Camera always looks at face center
        R_cam = build_camera_rotation(p_cam, face_center_world, roll_rad=0.0)
        q_cam = rotation_to_quaternion(R_cam)

        # Teleport camera
        req = SetModelStateRequest()
        req.model_state.model_name = "labeler_camera"
        req.model_state.reference_frame = "world"
        req.model_state.pose.position.x = float(p_cam[0])
        req.model_state.pose.position.y = float(p_cam[1])
        req.model_state.pose.position.z = float(p_cam[2])
        req.model_state.pose.orientation.x = float(q_cam[0])
        req.model_state.pose.orientation.y = float(q_cam[1])
        req.model_state.pose.orientation.z = float(q_cam[2])
        req.model_state.pose.orientation.w = float(q_cam[3])
        set_state(req)

        # Wait for render
        rospy.sleep(0.15)

        # Flush stale frame
        try:
            rospy.wait_for_message("/labeler_camera/image_raw", Image, timeout=1.0)
        except rospy.ROSException:
            pass

        # Capture fresh frame
        try:
            img_msg = rospy.wait_for_message("/labeler_camera/image_raw", Image, timeout=3.0)
        except rospy.ROSException:
            rospy.logwarn(f"Pose {idx + 1}: timed out, skipping.")
            continue

        # Project 3D valve circle → 2D pixels
        pixels = project_3d_to_2d(valve_circle_world, R_cam, p_cam)

        if pixels is None:
            rospy.logdebug(f"Pose {idx + 1}: valve behind camera, skipping label.")
            continue

        # Generate YOLO segmentation label
        label_str = pixels_to_yolo_seg(pixels, IMG_W, IMG_H, class_id=0, min_area_px=200)

        if label_str is None:
            rospy.logdebug(f"Pose {idx + 1}: valve too small or outside image, skipping.")
            continue

        # Determine split
        split = "valid" if idx in valid_set else "train"
        stem = f"valve_{image_count:05d}"

        # Save image
        img_path = os.path.join(output_dir, split, "images", stem + ".jpg")
        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv2.imwrite(img_path, cv_img)

        # Save label
        lbl_path = os.path.join(output_dir, split, "labels", stem + ".txt")
        with open(lbl_path, "w") as f:
            f.write(label_str + "\n")

        image_count += 1
        label_count += 1
        rospy.loginfo(f"[{split}] {image_count}/{total}  r={r}m θ={polar_deg}° φ={az_deg}°")

    # Clean up camera
    try:
        delete_proxy("labeler_camera")
    except Exception:
        pass

    # Write data.yaml for YOLOv8
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("nc: 1\n")
        f.write("names:\n")
        f.write("  - Valve_edge\n")

    rospy.loginfo(f"\n{'='*60}")
    rospy.loginfo(f"Done! Generated {image_count} images with {label_count} auto-labels.")
    rospy.loginfo(f"Dataset: {output_dir}")
    rospy.loginfo(f"data.yaml: {yaml_path}")
    rospy.loginfo(f"Ready for: yolo task=segment mode=train data={yaml_path}")
    rospy.loginfo(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
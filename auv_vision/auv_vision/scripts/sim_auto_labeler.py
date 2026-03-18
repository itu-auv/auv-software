
#!/usr/bin/env python3
"""
Auto-labeler for YOLO-pose valve training data.

Spawns a free-floating camera in Gazebo, teleports it through a hemisphere
grid of poses around valve_front, captures one image per pose, and writes
YOLO-pose keypoint labels. Because we set the camera pose directly, there
is no timestamp synchronization problem.

Usage:
    rosrun auv_vision sim_auto_labeler.py
    rosrun auv_vision sim_auto_labeler.py _output_dir:=~/my_dataset
"""

import os
import glob
import itertools
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, SetModelStateRequest
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_matrix, euler_matrix, quaternion_matrix


# ---------------------------------------------------------------------------
# Bolt hole centers on the subsea valve flange face, in STL coordinate system
# (meters — original STEP values are in mm, divided by 1000).
# X = 0.020 m is the flange face surface; bolt circle radius = 0.103 m.
# ---------------------------------------------------------------------------
BOLT_HOLES_STL_M = np.array([
    [0.020, -0.072832, -0.072832],  # Hole 1  -135°
    [0.020,  0.000000, -0.103000],  # Hole 2   -90°
    [0.020,  0.072832, -0.072832],  # Hole 3   -45°
    [0.020,  0.103000,  0.000000],  # Hole 4     0°
    [0.020,  0.072832,  0.072832],  # Hole 5    45°
    [0.020,  0.000000,  0.103000],  # Hole 6    90°
    [0.020, -0.072832,  0.072832],  # Hole 7   135°
    [0.020, -0.103000,  0.000000],  # Hole 8   180°
    [0.000, 0.000, 0.000], # Center Hole
    [0.000, 0.000235, -0.067256] # Arrow Tip
])

# valve_right (right panel) — face normal ends up as -X in world frame.
VALVE_LINK = {"pos": [0.58, 0.555, 1.4205], "rpy": [0.0, np.pi, 0.0]}

N_KEYPOINTS = len(BOLT_HOLES_STL_M)
CLASS_ID    = 0

# Flange face center in world frame (derived analytically).
# tac_valve model at world (7,-4,-3), valve_right link at model (0.58,0.555,1.4205),
# rpy (0,π,0) maps STL face offset [0.020,0,0] → desk [-0.020,0,0].
FACE_CENTER_WORLD = np.array([7.56, -3.445, -1.5795])

# ---------------------------------------------------------------------------
# Pose grid — hemisphere centered on the face normal direction.
# ---------------------------------------------------------------------------
RADII_M         = [0.3, 0.5, 0.75, 1.0, 1.5]
POLAR_DEG       = [0, 15, 40, 65, 80]       # polar angle from face normal
AZIMUTH_DEG     = list(range(0, 360, 45))   # 8 azimuths
ROLL_DEG        = [0, -30, 30]
ORIENTATIONS    = ["facing", "perpendicular"]

# Face normal in model frame: STL face normal [1,0,0] rotated by valve link rpy.
# (model has no rotation in world, so model frame = world frame orientation.)
_R_valve = euler_matrix(*VALVE_LINK["rpy"], axes="sxyz")[:3, :3]
FACE_NORMAL = _R_valve @ np.array([1.0, 0.0, 0.0])

# Orthonormal hemisphere frame: pole along face normal.
_ref = np.array([0.0, 0.0, 1.0]) if abs(FACE_NORMAL[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
_HEMI_E1 = np.cross(FACE_NORMAL, _ref); _HEMI_E1 /= np.linalg.norm(_HEMI_E1)
_HEMI_E2 = np.cross(_HEMI_E1, FACE_NORMAL)

# ---------------------------------------------------------------------------
# Standalone camera SDF — matches AUV front camera intrinsics exactly.
# Camera renders along its link's +X axis (Gazebo convention).
# No noise added so labels are pixel-perfect.
# ---------------------------------------------------------------------------
_CAMERA_SDF = """<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='labeler_camera'>
    <static>true</static>
    <link name='link'>
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

# Optical axis-swap rotation: camera body +X (Gazebo look-direction)
# → camera optical +Z (ROS convention, needed for pinhole projection with K).
# Matches the URDF optical joint: rpy="-π/2  0  -π/2".
R_OPT = euler_matrix(-np.pi / 2, 0.0, -np.pi / 2, axes="sxyz")[:3, :3]




# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def build_camera_rotation(p_cam, face_center, roll_rad):
    """Return 3×3 rotation matrix R such that R @ [1,0,0] = toward face_center.

    Camera body convention: +X forward (Gazebo), +Y left, +Z up in body.
    Roll is applied around the forward (+X) axis.
    """
    fwd = _normalize(face_center - p_cam)

    # Pick a reference "up" that avoids parallel-with-fwd singularity.
    ref = np.array([0.0, 0.0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1.0, 0.0, 0.0])

    left = _normalize(np.cross(ref, fwd))
    up   = np.cross(fwd, left)

    # R columns: body X, Y, Z expressed in world frame.
    R = np.column_stack([fwd, left, up])

    # Roll around body +X.
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    Rx = np.array([[1, 0,   0  ],
                   [0, cr, -sr ],
                   [0, sr,  cr ]])
    return R @ Rx


def rotation_to_quaternion(R):
    """Convert 3×3 rotation matrix to (x,y,z,w) quaternion."""
    M = np.eye(4)
    M[:3, :3] = R
    return quaternion_from_matrix(M)


def get_valve_keypoints_world(valve_model_pose):
    """Transform 8 bolt holes: STL frame → valve_front link frame → world frame."""
    p = valve_model_pose.position
    q = valve_model_pose.orientation
    R_model = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
    t_model = np.array([p.x, p.y, p.z])

    roll, pitch, yaw = VALVE_LINK["rpy"]
    R_link = euler_matrix(roll, pitch, yaw, axes="sxyz")[:3, :3]
    t_link = np.array(VALVE_LINK["pos"])

    p_link  = (R_link @ BOLT_HOLES_STL_M.T).T + t_link
    p_world = (R_model @ p_link.T).T + t_model
    return p_world


def world_to_camera(world_points, p_cam, R_body):
    """Transform world-frame 3D points into camera optical frame.

    p_cam  : (3,) camera position in world
    R_body : (3,3) rotation matrix (columns = camera body axes in world)

    Step 1: world → camera body  (R_body.T @ (P - p_cam))
    Step 2: camera body → optical  (R_OPT @)
    """
    P_body = (R_body.T @ (world_points - p_cam).T).T
    P_opt  = (R_OPT.T @ P_body.T).T
    return P_opt


def project_keypoints(cam_points, K, img_w, img_h):
    """Pinhole projection → list of (u, v, vis)."""
    fx, fy = K[0], K[4]
    cx, cy = K[2], K[5]
    result = []
    for pt in cam_points:
        if pt[2] <= 0.05:
            result.append((0.0, 0.0, 0))
        else:
            u = fx * pt[0] / pt[2] + cx
            v = fy * pt[1] / pt[2] + cy
            vis = 2 if (0 <= u < img_w and 0 <= v < img_h) else 0
            result.append((u, v, vis))
    return result


# ---------------------------------------------------------------------------
# Main labeler
# ---------------------------------------------------------------------------

def main():
    rospy.init_node("sim_auto_labeler")

    output_dir = rospy.get_param("~output_dir", os.path.expanduser("~/yolo_valve_dataset"))
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Clean previous dataset files
    for old in glob.glob(os.path.join(images_dir, "*.jpg")):
        os.remove(old)
    for old in glob.glob(os.path.join(labels_dir, "*.txt")):
        os.remove(old)
    rospy.loginfo("Cleared previous labels and images.")

    bridge = CvBridge()

    # ------------------------------------------------------------------
    # Gazebo services
    # ------------------------------------------------------------------
    rospy.loginfo("Waiting for Gazebo services...")
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    rospy.wait_for_service("/gazebo/delete_model")
    rospy.wait_for_service("/gazebo/set_model_state")

    spawn_proxy  = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    delete_proxy = rospy.ServiceProxy("/gazebo/delete_model",    DeleteModel)
    set_state    = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    # ------------------------------------------------------------------
    # Spawn labeler camera
    # ------------------------------------------------------------------
    rospy.loginfo("Spawning labeler_camera...")
    spawn_proxy(
        model_name="labeler_camera",
        model_xml=_CAMERA_SDF,
        robot_namespace="",
        initial_pose=Pose(),
        reference_frame="world",
    )
    rospy.sleep(1.0)   # allow Gazebo to register the camera publisher

    # Persistent image subscriber — keeps the camera rendering continuously
    # so that after teleporting + sleeping, the latest frame is always fresh.
    latest_image = [None]
    def _img_cb(msg):
        latest_image[0] = msg
    img_sub = rospy.Subscriber("/labeler_camera/image_raw", Image, _img_cb, queue_size=1)

    # ------------------------------------------------------------------
    # Get camera intrinsics
    # ------------------------------------------------------------------
    rospy.loginfo("Waiting for camera_info...")
    cam_info = rospy.wait_for_message("/labeler_camera/camera_info", CameraInfo, timeout=10.0)
    K     = np.array(cam_info.K)
    img_w = cam_info.width
    img_h = cam_info.height
    rospy.loginfo(f"Camera intrinsics: {img_w}×{img_h}, fx={K[0]:.1f}")

    # ------------------------------------------------------------------
    # Read valve pose once (it is static)
    # ------------------------------------------------------------------
    rospy.loginfo("Reading valve pose from model_states...")
    valve_pose = None
    def _ms_cb(msg):
        nonlocal valve_pose
        for i, name in enumerate(msg.name):
            if name == "tac_valve":
                valve_pose = msg.pose[i]
    sub = rospy.Subscriber("/gazebo/model_states", ModelStates, _ms_cb)
    deadline = rospy.Time.now() + rospy.Duration(5.0)
    while valve_pose is None and rospy.Time.now() < deadline:
        rospy.sleep(0.05)
    sub.unregister()
    if valve_pose is None:
        rospy.logerr("Could not get tac_valve pose — is the simulation running with tac_sea.world?")
        return
    rospy.loginfo(f"Valve pose acquired: pos=({valve_pose.position.x:.3f}, "
                  f"{valve_pose.position.y:.3f}, {valve_pose.position.z:.3f})")

    # ------------------------------------------------------------------
    # Build pose grid
    # ------------------------------------------------------------------
    poses = []
    for combo in itertools.product(RADII_M, POLAR_DEG, AZIMUTH_DEG, ROLL_DEG, ORIENTATIONS):
        r, polar_deg, az_deg, roll_deg, orient = combo
        if polar_deg == 0:
            # Directly ahead: all azimuths give the same position,
            # and roll just rotates the image, so keep only one.
            if az_deg != 0 or roll_deg != 0:
                continue
        poses.append(combo)
    total = len(poses)
    rospy.loginfo(f"Grid: {total} poses — starting capture...")

    # Precompute valve keypoints (static)
    world_kps = get_valve_keypoints_world(valve_pose)

    image_count = 0

    for idx, (r, polar_deg, az_deg, roll_deg, orient) in enumerate(poses):
        if rospy.is_shutdown():
            break

        theta = np.radians(polar_deg)
        phi   = np.radians(az_deg)
        roll  = np.radians(roll_deg)

        # Camera position on hemisphere (pole = face normal)
        p_cam = FACE_CENTER_WORLD + r * (
            np.cos(theta) * FACE_NORMAL +
            np.sin(theta) * (np.cos(phi) * _HEMI_E1 + np.sin(phi) * _HEMI_E2)
        )

        if orient == "facing":
            # Camera looks toward the valve face center
            R_body = build_camera_rotation(p_cam, FACE_CENTER_WORLD, roll)
        else:
            # Camera looks along -FACE_NORMAL (perpendicular to face)
            perp_target = p_cam - FACE_NORMAL
            R_body = build_camera_rotation(p_cam, perp_target, roll)

        q = rotation_to_quaternion(R_body)

        # Teleport camera
        req = SetModelStateRequest()
        req.model_state.model_name      = "labeler_camera"
        req.model_state.reference_frame = "world"
        req.model_state.pose.position.x = float(p_cam[0])
        req.model_state.pose.position.y = float(p_cam[1])
        req.model_state.pose.position.z = float(p_cam[2])
        req.model_state.pose.orientation.x = float(q[0])
        req.model_state.pose.orientation.y = float(q[1])
        req.model_state.pose.orientation.z = float(q[2])
        req.model_state.pose.orientation.w = float(q[3])
        set_state(req)

        # Wait for the camera to render several frames at the new pose
        rospy.sleep(0.5)
        img_msg = latest_image[0]
        if img_msg is None:
            rospy.logwarn(f"Pose {idx+1}/{total}: no image received yet, skipping.")
            continue

        # Project keypoints
        cam_kps = world_to_camera(world_kps, p_cam, R_body)
        kp_img  = project_keypoints(cam_kps, K, img_w, img_h)
        n_vis   = sum(1 for _, _, v in kp_img if v == 2)

        if n_vis == 0:
            rospy.loginfo(f"Pose {idx+1}/{total}: 0 keypoints visible, skipping.")
            continue

        # Bounding box from visible keypoints
        visible = [(u, v) for u, v, vis in kp_img if vis == 2]
        us, vs  = zip(*visible)
        x_min = max(0, min(us));  x_max = min(img_w, max(us))
        y_min = max(0, min(vs));  y_max = min(img_h, max(vs))

        x_c = ((x_min + x_max) / 2) / img_w
        y_c = ((y_min + y_max) / 2) / img_h
        bw  = (x_max - x_min) / img_w
        bh  = (y_max - y_min) / img_h

        # Build YOLO-pose label line
        kp_fields = []
        for u, v, vis in kp_img:
            kp_fields += [f"{u/img_w:.6f}", f"{v/img_h:.6f}", str(vis)]
        label_line = f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_fields)

        # Save image and label
        stem      = f"valve_{image_count:05d}"
        img_path  = os.path.join(images_dir, stem + ".jpg")
        lbl_path  = os.path.join(labels_dir, stem + ".txt")

        cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        cv2.imwrite(img_path, cv_img)
        with open(lbl_path, "w") as f:
            f.write(label_line + "\n")

        image_count += 1
        rospy.loginfo(f"Image {image_count} (pose {idx+1}/{total}): "
                      f"r={r}m θ={polar_deg}° φ={az_deg}° roll={roll_deg}° {orient} "
                      f"— {n_vis}/{N_KEYPOINTS} keypoints visible")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    rospy.loginfo(f"Done. {image_count} images saved to {output_dir}")

    img_sub.unregister()
    try:
        delete_proxy("labeler_camera")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass


#!/usr/bin/env python3

import importlib.util
import math
import sys
import threading
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


def _install_ros_stubs():
    rospy = types.ModuleType("rospy")
    rospy.logwarn_throttle = lambda *args, **kwargs: None
    rospy.logdebug_throttle = lambda *args, **kwargs: None
    rospy.loginfo = lambda *args, **kwargs: None
    rospy.init_node = lambda *args, **kwargs: None
    rospy.get_param = lambda _name, default=None: default
    rospy.Subscriber = lambda *args, **kwargs: None
    rospy.Publisher = lambda *args, **kwargs: None
    rospy.Time = types.SimpleNamespace(now=lambda: 0.0)
    rospy.Duration = lambda *args, **kwargs: None
    rospy.ROSInterruptException = Exception
    rospy.spin = lambda: None
    sys.modules["rospy"] = rospy

    message_filters = types.ModuleType("message_filters")
    message_filters.Subscriber = lambda *args, **kwargs: None

    class _ApproximateTimeSynchronizer:
        def __init__(self, *args, **kwargs):
            pass

        def registerCallback(self, _callback):
            pass

    message_filters.ApproximateTimeSynchronizer = _ApproximateTimeSynchronizer
    sys.modules["message_filters"] = message_filters

    tf2_ros = types.ModuleType("tf2_ros")
    tf2_ros.TransformBroadcaster = lambda *args, **kwargs: None
    sys.modules["tf2_ros"] = tf2_ros

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    cv_bridge.CvBridgeError = Exception
    sys.modules["cv_bridge"] = cv_bridge

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.TransformStamped = object
    sys.modules["geometry_msgs"] = geometry_msgs
    sys.modules["geometry_msgs.msg"] = geometry_msgs_msg

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = object
    sys.modules["nav_msgs"] = nav_msgs
    sys.modules["nav_msgs.msg"] = nav_msgs_msg

    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg.CameraInfo = object
    sensor_msgs_msg.Image = object
    sys.modules["sensor_msgs"] = sensor_msgs
    sys.modules["sensor_msgs.msg"] = sensor_msgs_msg

    tf = types.ModuleType("tf")
    tf_transformations = types.ModuleType("tf.transformations")
    tf_transformations.euler_from_quaternion = lambda _q: (0.0, 0.0, 0.0)
    tf_transformations.quaternion_from_euler = lambda _r, _p, _y: (0.0, 0.0, 0.0, 1.0)
    sys.modules["tf"] = tf
    sys.modules["tf.transformations"] = tf_transformations


def _load_module():
    _install_ros_stubs()
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dynamic_roll_compensation.py"
    )
    spec = importlib.util.spec_from_file_location("dynamic_roll_compensation", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


drc = _load_module()


class _FakePublisher:
    def __init__(self):
        self.messages = []

    def get_num_connections(self):
        return 1

    def publish(self, msg):
        self.messages.append(msg)


class _FakeHeader:
    def __init__(self):
        self.stamp = 123.0
        self.seq = 7
        self.frame_id = "camera_optical"


class _FakeRoi:
    def __init__(self):
        self.x_offset = 0
        self.y_offset = 0
        self.width = 0
        self.height = 0
        self.do_rectify = False


class _FakeCameraInfo:
    def __init__(self):
        self.header = _FakeHeader()
        self.width = 640
        self.height = 480
        self.K = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
        self.P = [500.0, 0.0, 320.0, 0.0, 0.0, 500.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.roi = _FakeRoi()


class _FakeImage:
    def __init__(self):
        self.header = _FakeHeader()
        self.width = 640
        self.height = 480


class DynamicRollCompensationTest(unittest.TestCase):
    def _new_stabilizer_shell(self):
        node = drc.CameraRollStabilizer.__new__(drc.CameraRollStabilizer)
        node.crop_to_valid_region = True
        node.border_mode = cv2.BORDER_CONSTANT
        node.info_pub = _FakePublisher()
        node.lock = threading.Lock()
        node.camera_info = _FakeCameraInfo()
        node.camera_optical_frame_stabilized = "camera_optical_stabilized"
        return node

    def test_centered_valid_crop_stays_inside_rotated_dummy_image(self):
        node = self._new_stabilizer_shell()
        width, height = 160, 120
        angle_rad = math.radians(25.0)
        angle_deg = math.degrees(angle_rad)
        center = (width * 0.5, height * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        crop_rect = node._compute_valid_crop_rect(width, height, center, angle_rad)
        x0, y0, x1, y1 = crop_rect

        self.assertGreater(x1 - x0, 0)
        self.assertGreater(y1 - y0, 0)
        self.assertLess(x1 - x0, width)
        self.assertLess(y1 - y0, height)

        dummy = np.full((height, width, 3), 255, dtype=np.uint8)
        rotated = cv2.warpAffine(
            dummy,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cropped = rotated[y0:y1, x0:x1]

        self.assertTrue(np.all(cropped > 0))

    def test_off_center_valid_crop_stays_inside_rotated_image(self):
        node = self._new_stabilizer_shell()
        width, height = 160, 120
        angle_rad = math.radians(-17.0)
        angle_deg = math.degrees(angle_rad)
        center = (82.5, 58.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        crop_rect = node._compute_valid_crop_rect(width, height, center, angle_rad)
        x0, y0, x1, y1 = crop_rect

        valid_source = np.full((height, width), 255, dtype=np.uint8)
        valid_rotated = cv2.warpAffine(
            valid_source,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        self.assertGreater(x1 - x0, 0)
        self.assertGreater(y1 - y0, 0)
        self.assertTrue(np.all(valid_rotated[y0:y1, x0:x1] == 255))

    def test_stabilized_camera_info_shifts_principal_point_for_crop(self):
        node = self._new_stabilizer_shell()
        img_msg = _FakeImage()
        crop_rect = (13, 21, 613, 421)

        node._publish_stabilized_camera_info(img_msg, crop_rect)

        self.assertEqual(len(node.info_pub.messages), 1)
        stabilized = node.info_pub.messages[0]

        self.assertEqual(stabilized.header.stamp, img_msg.header.stamp)
        self.assertEqual(stabilized.header.seq, img_msg.header.seq)
        self.assertEqual(stabilized.header.frame_id, "camera_optical_stabilized")
        self.assertEqual(stabilized.width, 600)
        self.assertEqual(stabilized.height, 400)
        self.assertEqual(stabilized.K[0], 500.0)
        self.assertEqual(stabilized.K[4], 500.0)
        self.assertEqual(stabilized.K[2], 307.0)
        self.assertEqual(stabilized.K[5], 219.0)
        self.assertEqual(stabilized.P[2], 307.0)
        self.assertEqual(stabilized.P[6], 219.0)
        self.assertEqual(stabilized.roi.width, 600)
        self.assertEqual(stabilized.roi.height, 400)

    def test_rotation_center_prefers_matching_camera_info_principal_point(self):
        node = self._new_stabilizer_shell()

        self.assertEqual(node._rotation_center(640, 480), (320.0, 240.0))
        self.assertEqual(node._rotation_center(320, 240), (160.0, 120.0))


if __name__ == "__main__":
    unittest.main()

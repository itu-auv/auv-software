#!/usr/bin/env python3

import math
import importlib.util
import os
import sys
import types
import unittest


class Header:
    def __init__(self):
        self.frame_id = "odom"
        self.stamp = None


class Point:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Quaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()


class PoseStamped:
    def __init__(self):
        self.header = Header()
        self.pose = Pose()


class Path:
    def __init__(self):
        self.header = Header()
        self.poses = []


class Transform:
    def __init__(self):
        self.translation = Point()
        self.rotation = Quaternion()


class TransformStamped:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.transform = Transform()


class Time:
    def __init__(self, _=0.0):
        pass


class Duration:
    def __init__(self, _=0.0):
        pass


def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def euler_from_quaternion(quaternion):
    x, y, z, w = quaternion
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def install_ros_stubs():
    rospy = types.ModuleType("rospy")
    rospy.Time = Time
    rospy.Duration = Duration
    rospy.logerr = lambda *args, **kwargs: None
    sys.modules.setdefault("rospy", rospy)

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Path = Path
    nav_msgs.msg = nav_msgs_msg
    sys.modules.setdefault("nav_msgs", nav_msgs)
    sys.modules.setdefault("nav_msgs.msg", nav_msgs_msg)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseStamped = PoseStamped
    geometry_msgs_msg.TransformStamped = TransformStamped
    geometry_msgs.msg = geometry_msgs_msg
    sys.modules.setdefault("geometry_msgs", geometry_msgs)
    sys.modules.setdefault("geometry_msgs.msg", geometry_msgs_msg)

    tf2_ros = types.ModuleType("tf2_ros")
    tf2_ros.LookupException = Exception
    tf2_ros.ConnectivityException = Exception
    tf2_ros.ExtrapolationException = Exception
    tf2_ros.Buffer = object
    tf2_ros.TransformBroadcaster = object
    sys.modules.setdefault("tf2_ros", tf2_ros)

    tf2_geometry_msgs = types.ModuleType("tf2_geometry_msgs")
    tf2_geometry_msgs.do_transform_pose = lambda pose, _: pose
    sys.modules.setdefault("tf2_geometry_msgs", tf2_geometry_msgs)

    tf = types.ModuleType("tf")
    transformations = types.ModuleType("tf.transformations")
    transformations.euler_from_quaternion = euler_from_quaternion
    transformations.quaternion_from_euler = quaternion_from_euler
    tf.transformations = transformations
    sys.modules.setdefault("tf", tf)
    sys.modules.setdefault("tf.transformations", transformations)


install_ros_stubs()
HELPERS_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src",
        "auv_navigation",
        "follow_path_action",
        "follow_path_helpers.py",
    )
)

spec = importlib.util.spec_from_file_location("follow_path_helpers", HELPERS_PATH)
follow_path_helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow_path_helpers)


def make_pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    pose = PoseStamped()
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    quaternion = quaternion_from_euler(0.0, 0.0, yaw)
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    return pose


class TestYawGate(unittest.TestCase):
    def assert_yaw_close(self, pose, expected_yaw):
        actual_yaw = follow_path_helpers.get_pose_yaw(pose)
        diff = follow_path_helpers.normalize_angle(actual_yaw - expected_yaw)
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_small_yaw_error_keeps_regular_target(self):
        target = make_pose(3.0, 4.0, yaw=0.1)
        robot = make_pose(0.0, 0.0, yaw=0.0)

        gated, yaw_gate_active = follow_path_helpers.apply_yaw_gate(
            target, robot, False, 0.4, 0.2, 0.2
        )

        self.assertFalse(yaw_gate_active)
        self.assertAlmostEqual(gated.pose.position.x, 3.0)
        self.assertAlmostEqual(gated.pose.position.y, 4.0)
        self.assert_yaw_close(gated, 0.1)

    def test_large_yaw_error_enters_yaw_gate_with_position_lookahead(self):
        target = make_pose(5.0, 3.0, yaw=1.2)
        robot = make_pose(2.0, 3.0, yaw=0.0)

        gated, yaw_gate_active = follow_path_helpers.apply_yaw_gate(
            target, robot, False, 0.4, 0.2, 0.2
        )

        self.assertTrue(yaw_gate_active)
        self.assertAlmostEqual(gated.pose.position.x, 2.2)
        self.assertAlmostEqual(gated.pose.position.y, 3.0)
        self.assert_yaw_close(gated, 1.2)

    def test_yaw_gate_keeps_position_target_near_robot_until_yaw_is_close(self):
        target = make_pose(5.0, 9.0, yaw=1.1)
        robot = make_pose(9.0, 9.0, yaw=0.2)

        gated, yaw_gate_active = follow_path_helpers.apply_yaw_gate(
            target, robot, True, 0.4, 0.2, 0.2
        )

        self.assertTrue(yaw_gate_active)
        self.assertAlmostEqual(gated.pose.position.x, 8.8)
        self.assertAlmostEqual(gated.pose.position.y, 9.0)
        self.assert_yaw_close(gated, 1.1)

    def test_yaw_gate_exits_when_yaw_is_close(self):
        target = make_pose(5.0, 6.0, yaw=1.1)
        robot = make_pose(9.0, 9.0, yaw=1.05)

        gated, yaw_gate_active = follow_path_helpers.apply_yaw_gate(
            target, robot, True, 0.4, 0.2, 0.2
        )

        self.assertFalse(yaw_gate_active)
        self.assertAlmostEqual(gated.pose.position.x, 5.0)
        self.assertAlmostEqual(gated.pose.position.y, 6.0)
        self.assert_yaw_close(gated, 1.1)

    def test_wraparound_yaw_error_does_not_trigger_yaw_gate(self):
        target = make_pose(5.0, 6.0, yaw=-math.pi + 0.05)
        robot = make_pose(0.0, 0.0, yaw=math.pi - 0.05)

        gated, yaw_gate_active = follow_path_helpers.apply_yaw_gate(
            target, robot, False, 0.4, 0.2, 0.2
        )

        self.assertFalse(yaw_gate_active)
        self.assert_yaw_close(gated, -math.pi + 0.05)

    def test_zero_position_lookahead_holds_current_position(self):
        target = make_pose(5.0, 3.0, yaw=1.2)
        robot = make_pose(2.0, 3.0, yaw=0.0)

        gated, yaw_gate_active = follow_path_helpers.apply_yaw_gate(
            target, robot, False, 0.4, 0.2, 0.0
        )

        self.assertTrue(yaw_gate_active)
        self.assertAlmostEqual(gated.pose.position.x, 2.0)
        self.assertAlmostEqual(gated.pose.position.y, 3.0)


if __name__ == "__main__":
    unittest.main()

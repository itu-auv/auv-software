#!/usr/bin/env python3

import importlib.util
import os
import sys
import types
import unittest


def _install_import_stubs():
    rospy = types.ModuleType("rospy")
    rospy.Time = object
    rospy.Duration = object
    sys.modules.setdefault("rospy", rospy)

    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.Twist = object
    geometry_msgs_msg.WrenchStamped = object
    sys.modules.setdefault("geometry_msgs", geometry_msgs)
    sys.modules.setdefault("geometry_msgs.msg", geometry_msgs_msg)

    std_msgs = types.ModuleType("std_msgs")
    std_msgs_msg = types.ModuleType("std_msgs.msg")
    std_msgs_msg.Bool = object
    sys.modules.setdefault("std_msgs", std_msgs)
    sys.modules.setdefault("std_msgs.msg", std_msgs_msg)

    std_srvs = types.ModuleType("std_srvs")
    std_srvs_srv = types.ModuleType("std_srvs.srv")
    std_srvs_srv.SetBool = object
    std_srvs_srv.SetBoolResponse = object
    sys.modules.setdefault("std_srvs", std_srvs)
    sys.modules.setdefault("std_srvs.srv", std_srvs_srv)

    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    nav_msgs_msg.Odometry = object
    sys.modules.setdefault("nav_msgs", nav_msgs)
    sys.modules.setdefault("nav_msgs.msg", nav_msgs_msg)

    message_filters = types.ModuleType("message_filters")
    sys.modules.setdefault("message_filters", message_filters)

    tf = types.ModuleType("tf")
    tf_transformations = types.ModuleType("tf.transformations")
    tf.transformations = tf_transformations
    sys.modules.setdefault("tf", tf)
    sys.modules.setdefault("tf.transformations", tf_transformations)

    auv_common_lib = types.ModuleType("auv_common_lib")
    logging = types.ModuleType("auv_common_lib.logging")
    terminal_color_utils = types.ModuleType(
        "auv_common_lib.logging.terminal_color_utils"
    )

    class TerminalColors:
        PASTEL_BLUE = ""
        PASTEL_GREEN = ""
        PASTEL_RED = ""

        @staticmethod
        def color_text(text, _color):
            return text

    terminal_color_utils.TerminalColors = TerminalColors
    sys.modules.setdefault("auv_common_lib", auv_common_lib)
    sys.modules.setdefault("auv_common_lib.logging", logging)
    sys.modules.setdefault(
        "auv_common_lib.logging.terminal_color_utils", terminal_color_utils
    )


def _load_dvl_module():
    _install_import_stubs()
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_path = os.path.join(package_dir, "scripts", "dvl_odometry_node.py")
    spec = importlib.util.spec_from_file_location("dvl_odometry_node", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DvlValidityRollingStats = _load_dvl_module().DvlValidityRollingStats


class TestDvlValidityRollingStats(unittest.TestCase):
    def test_reports_invalid_per_100_for_current_window(self):
        stats = DvlValidityRollingStats(window_duration_sec=60.0)

        for i in range(100):
            stats.record(stamp_sec=i * 0.1, is_valid=i >= 25)

        total_count, invalid_count, invalid_per_100 = stats.summary(now_sec=10.0)

        self.assertEqual(total_count, 100)
        self.assertEqual(invalid_count, 25)
        self.assertAlmostEqual(invalid_per_100, 25.0)

    def test_prunes_old_samples_and_updates_invalid_count(self):
        stats = DvlValidityRollingStats(window_duration_sec=5.0)
        stats.record(stamp_sec=0.0, is_valid=False)
        stats.record(stamp_sec=4.0, is_valid=True)
        stats.record(stamp_sec=6.0, is_valid=True)

        total_count, invalid_count, invalid_per_100 = stats.summary(now_sec=6.0)

        self.assertEqual(total_count, 2)
        self.assertEqual(invalid_count, 0)
        self.assertAlmostEqual(invalid_per_100, 0.0)

    def test_keeps_samples_on_exact_window_boundary(self):
        stats = DvlValidityRollingStats(window_duration_sec=5.0)
        stats.record(stamp_sec=0.0, is_valid=False)
        stats.record(stamp_sec=5.0, is_valid=True)

        total_count, invalid_count, invalid_per_100 = stats.summary(now_sec=5.0)

        self.assertEqual(total_count, 2)
        self.assertEqual(invalid_count, 1)
        self.assertAlmostEqual(invalid_per_100, 50.0)

    def test_empty_summary_returns_none(self):
        stats = DvlValidityRollingStats(window_duration_sec=60.0)

        self.assertIsNone(stats.summary(now_sec=0.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

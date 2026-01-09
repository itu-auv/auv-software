#!/usr/bin/env python3
"""
Integration test for slalom visual servoing pipeline.

Tests the flow: ObjectDetectionArray -> VS Controller -> cmd_vel

Usage:
    rosrun auv_control test_slalom_integration.py

Publishes fake pipe detections and verifies controller responds correctly.
"""

import rospy
from geometry_msgs.msg import Point, Twist
from std_srvs.srv import Trigger
from auv_msgs.msg import ObjectDetection, ObjectDetectionArray
from auv_msgs.srv import VisualServoing


class SlalomIntegrationTest:
    def __init__(self):
        rospy.init_node("test_slalom_integration")

        self.cmd_vel_received = None
        self.test_results = []

        # Publisher for fake detections
        self.pipe_pub = rospy.Publisher(
            "slalom_pipes", ObjectDetectionArray, queue_size=1
        )

        # Subscriber to verify controller output
        rospy.Subscriber("cmd_vel", Twist, self._cmd_vel_cb, queue_size=1)

        # Wait for controller services
        rospy.loginfo("Waiting for visual_servoing services...")
        rospy.wait_for_service("visual_servoing/start", timeout=10.0)
        rospy.wait_for_service("visual_servoing/cancel", timeout=10.0)

        self.start_srv = rospy.ServiceProxy("visual_servoing/start", VisualServoing)
        self.cancel_srv = rospy.ServiceProxy("visual_servoing/cancel", Trigger)

        rospy.loginfo("Services connected.")

    def _cmd_vel_cb(self, msg: Twist):
        self.cmd_vel_received = msg

    def _create_detection(
        self, x: float, depth: float, color: str = "white"
    ) -> ObjectDetection:
        """Create a pipe detection at normalized x position."""
        det = ObjectDetection()
        det.label = "pipe"
        det.color = color
        det.confidence = 0.9
        det.depth = depth
        det.centroid = Point(x=x, y=0.0, z=0.0)
        det.bbox = [100, 100, 20, 200]
        return det

    def _publish_pipes(self, detections: list):
        """Publish an ObjectDetectionArray."""
        msg = ObjectDetectionArray()
        msg.header.stamp = rospy.Time.now()
        msg.detections = detections
        self.pipe_pub.publish(msg)

    def _wait_for_cmd_vel(self, timeout: float = 1.0) -> bool:
        """Wait for cmd_vel to be received."""
        self.cmd_vel_received = None
        start = rospy.Time.now()
        rate = rospy.Rate(50)
        while (rospy.Time.now() - start).to_sec() < timeout:
            if self.cmd_vel_received is not None:
                return True
            rate.sleep()
        return False

    def test_centered_pipes(self):
        """Test: Two pipes centered -> angular.z should be near zero."""
        rospy.loginfo("Test: Centered pipes")

        # Red on left, white on right, centered
        detections = [
            self._create_detection(x=-0.3, depth=2.0, color="red"),
            self._create_detection(x=0.3, depth=2.0, color="white"),
        ]

        self._publish_pipes(detections)
        rospy.sleep(0.2)
        self._publish_pipes(detections)

        if self._wait_for_cmd_vel():
            angular_z = self.cmd_vel_received.angular.z
            passed = abs(angular_z) < 0.2
            rospy.loginfo(
                f"  angular.z = {angular_z:.3f} (expect ~0) -> {'PASS' if passed else 'FAIL'}"
            )
            return passed
        else:
            rospy.logerr("  No cmd_vel received -> FAIL")
            return False

    def test_pipes_on_right(self):
        """Test: Pipes shifted right -> angular.z should be positive (turn left)."""
        rospy.loginfo("Test: Pipes on right")

        detections = [
            self._create_detection(x=0.2, depth=2.0, color="red"),
            self._create_detection(x=0.6, depth=2.0, color="white"),
        ]

        self._publish_pipes(detections)
        rospy.sleep(0.2)
        self._publish_pipes(detections)

        if self._wait_for_cmd_vel():
            angular_z = self.cmd_vel_received.angular.z
            passed = angular_z > 0.1
            rospy.loginfo(
                f"  angular.z = {angular_z:.3f} (expect > 0) -> {'PASS' if passed else 'FAIL'}"
            )
            return passed
        else:
            rospy.logerr("  No cmd_vel received -> FAIL")
            return False

    def test_pipes_on_left(self):
        """Test: Pipes shifted left -> angular.z should be negative (turn right)."""
        rospy.loginfo("Test: Pipes on left")

        detections = [
            self._create_detection(x=-0.6, depth=2.0, color="red"),
            self._create_detection(x=-0.2, depth=2.0, color="white"),
        ]

        self._publish_pipes(detections)
        rospy.sleep(0.2)
        self._publish_pipes(detections)

        if self._wait_for_cmd_vel():
            angular_z = self.cmd_vel_received.angular.z
            passed = angular_z < -0.1
            rospy.loginfo(
                f"  angular.z = {angular_z:.3f} (expect < 0) -> {'PASS' if passed else 'FAIL'}"
            )
            return passed
        else:
            rospy.logerr("  No cmd_vel received -> FAIL")
            return False

    def run_tests(self):
        """Run all integration tests."""
        rospy.loginfo("=" * 50)
        rospy.loginfo("SLALOM INTEGRATION TESTS")
        rospy.loginfo("=" * 50)

        # Start controller in slalom mode
        rospy.loginfo("Starting VS controller for 'slalom' target...")
        resp = self.start_srv(target_prop="slalom")
        if not resp.success:
            rospy.logerr(f"Failed to start controller: {resp.message}")
            return

        rospy.sleep(0.5)

        # Run tests
        results = [
            ("Centered pipes", self.test_centered_pipes()),
            ("Pipes on right", self.test_pipes_on_right()),
            ("Pipes on left", self.test_pipes_on_left()),
        ]

        # Cancel controller
        self.cancel_srv()

        # Summary
        rospy.loginfo("=" * 50)
        rospy.loginfo("RESULTS:")
        passed = sum(1 for _, r in results if r)
        for name, result in results:
            rospy.loginfo(f"  {name}: {'PASS' if result else 'FAIL'}")
        rospy.loginfo(f"Passed: {passed}/{len(results)}")
        rospy.loginfo("=" * 50)


if __name__ == "__main__":
    try:
        test = SlalomIntegrationTest()
        test.run_tests()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")

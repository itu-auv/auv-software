#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rospy
from auv_msgs.msg import SlalomProp, SlalomTarget


def _is_red(obj: str) -> bool:
    return "red" in obj.lower()


def _is_white(obj: str) -> bool:
    return "white" in obj.lower()


def _bbox_area(prop: SlalomProp) -> float:
    return max(0.0, prop.bbox_size_x) * max(0.0, prop.bbox_size_y)


class SlalomTargetPublisher:
    def __init__(self):
        rospy.init_node("slalom_target_publisher")

        self.direction = rospy.get_param("~direction", "left").lower()
        if self.direction not in ("left", "right"):
            rospy.logwarn(
                "Unsupported slalom direction '%s'. Falling back to 'left'.",
                self.direction,
            )
            self.direction = "left"

        self.frame_tolerance_s = rospy.get_param("~frame_tolerance_s", 0.05)
        self.min_angle_separation = rospy.get_param("~min_angle_separation_rad", 0.03)

        self.current_stamp: Optional[rospy.Time] = None
        self.red_props: List[SlalomProp] = []
        self.white_props: List[SlalomProp] = []
        self.last_published_key: Optional[Tuple[int, int, int, int]] = None

        self.target_pub = rospy.Publisher("slalom/target", SlalomTarget, queue_size=1)
        rospy.Subscriber("slalom/props", SlalomProp, self.prop_callback, queue_size=10)

        rospy.loginfo(
            "Slalom target publisher started. direction=%s min_angle_separation=%.3f",
            self.direction,
            self.min_angle_separation,
        )

    def prop_callback(self, msg: SlalomProp):
        if not (_is_red(msg.object) or _is_white(msg.object)):
            return

        if self._is_new_frame(msg.header.stamp):
            self._reset_frame(msg.header.stamp)

        if _is_red(msg.object):
            self.red_props.append(msg)
        else:
            self.white_props.append(msg)

        self._publish_if_ready(msg.header.stamp)

    def _is_new_frame(self, stamp: rospy.Time) -> bool:
        if self.current_stamp is None:
            return True
        return abs((stamp - self.current_stamp).to_sec()) > self.frame_tolerance_s

    def _reset_frame(self, stamp: rospy.Time):
        self.current_stamp = stamp
        self.red_props = []
        self.white_props = []
        self.last_published_key = None

    def _publish_if_ready(self, stamp: rospy.Time):
        red = self._closest(self.red_props)
        if red is None:
            return

        whites = [
            white
            for white in self.white_props
            if self._is_on_selected_side(red, white)
            and abs(white.angle - red.angle) >= self.min_angle_separation
        ]
        white = self._closest(whites)
        if white is None:
            rospy.logwarn_throttle(
                1.0,
                "No white slalom pipe on %s side of the closest red pipe.",
                self.direction,
            )
            return

        key = (
            stamp.secs,
            stamp.nsecs,
            int(red.bbox_center_x * 10.0),
            int(white.bbox_center_x * 10.0),
        )
        if key == self.last_published_key:
            return
        self.last_published_key = key

        target = SlalomTarget()
        target.header.stamp = stamp
        target.direction = self.direction
        target.red_angle = red.angle
        target.white_angle = white.angle
        target.yaw_error = 0.5 * (red.angle + white.angle)
        target.red_distance = red.distance
        target.white_distance = white.distance
        target.red_bbox_center_x = red.bbox_center_x
        target.white_bbox_center_x = white.bbox_center_x
        self.target_pub.publish(target)

    def _closest(self, props: List[SlalomProp]) -> Optional[SlalomProp]:
        if not props:
            return None

        def key(prop: SlalomProp):
            if math.isfinite(prop.distance) and prop.distance > 0.0:
                return (prop.distance, -_bbox_area(prop))
            return (float("inf"), -_bbox_area(prop))

        return min(props, key=key)

    def _is_on_selected_side(self, red: SlalomProp, white: SlalomProp) -> bool:
        if red.bbox_center_x > 0.0 and white.bbox_center_x > 0.0:
            if self.direction == "right":
                return white.bbox_center_x > red.bbox_center_x
            return white.bbox_center_x < red.bbox_center_x

        if self.direction == "right":
            return white.angle < red.angle
        return white.angle > red.angle

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        SlalomTargetPublisher().spin()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3
"""Publish laptop webcam frames as sensor_msgs/Image (+ a plausible CameraInfo).

The CameraInfo intrinsics are SYNTHETIC — a pinhole model from an assumed
horizontal FOV, zero distortion — sized to the actual capture resolution. Good
enough for nodes that need intrinsics (e.g. the valve keypoint node's PnP
consensus gate), NOT a real calibration.

Usage:
    rosrun auv_cam webcam_publisher.py \
        [_device:=0] \
        [_topic:=/taluy/cameras/cam_front/image_raw] \
        [_camera_info_topic:=/taluy/cameras/cam_front/camera_info] \
        [_rate:=30] [_hfov_deg:=60.0]
"""

import math

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


def build_camera_info(width, height, hfov_deg, frame_id):
    """Pinhole intrinsics from an assumed horizontal FOV (square pixels,
    principal point at centre, no distortion). SYNTHETIC — not a calibration."""
    fx = width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    info = CameraInfo()
    info.width = width
    info.height = height
    info.distortion_model = "plumb_bob"
    info.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    info.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    info.header.frame_id = frame_id
    return info


def main():
    rospy.init_node("webcam_publisher", anonymous=True)

    device = rospy.get_param("~device", 0)
    topic = rospy.get_param("~topic", "/taluy/cameras/cam_front/image_raw")
    camera_info_topic = rospy.get_param("~camera_info_topic", "")
    if not camera_info_topic:
        camera_info_topic = topic.rsplit("/", 1)[0] + "/camera_info"
    rate_hz = rospy.get_param("~rate", 30)
    hfov_deg = float(rospy.get_param("~hfov_deg", 60.0))

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        rospy.logfatal(f"Cannot open webcam device {device}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_id = "taluy/base_link/front_camera_optical_link"
    rospy.loginfo(
        f"Webcam opened: device={device}, {width}x{height}, publishing on {topic} "
        f"@ {rate_hz} Hz"
    )

    cam_info = build_camera_info(width, height, hfov_deg, frame_id)
    rospy.loginfo(
        "Synthetic CameraInfo on %s: hfov=%.1fdeg -> fx=fy=%.1f cx=%.1f cy=%.1f "
        "(NOT a real calibration)",
        camera_info_topic,
        hfov_deg,
        cam_info.K[0],
        cam_info.K[4],
        cam_info.K[5],
    )

    pub = rospy.Publisher(topic, Image, queue_size=1)
    info_pub = rospy.Publisher(camera_info_topic, CameraInfo, queue_size=1)
    bridge = CvBridge()
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn_throttle(5.0, "Failed to read frame from webcam")
            continue

        # Capture resolution can change after open; keep CameraInfo in sync.
        h, w = frame.shape[:2]
        if w != cam_info.width or h != cam_info.height:
            cam_info = build_camera_info(w, h, hfov_deg, frame_id)

        stamp = rospy.Time.now()
        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

        cam_info.header.stamp = stamp
        info_pub.publish(cam_info)
        rate.sleep()

    cap.release()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3

import rosbag
import rospy
import tf2_ros
import tf2_geometry_msgs
import numpy as np
import argparse
import os
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math


def get_bearing_angle(x, y):
    """
    Calculate the bearing angle (yaw) to a target point.
    This is the angle of the vector from origin to (x, y) in the XY plane.
    Returns angle in radians, where 0 = straight ahead (+X), positive = left (+Y).
    """
    return math.atan2(y, x)


def get_transforms_at_time(tf_buffer, target_frames, source_frame, time):
    """
    Queries the TF buffer for transforms from source_frame to each target_frame at a specific time.
    Returns a flattened list of [x, y, z, yaw] for each target frame.
    Note: Roll and pitch are omitted as they're typically controlled separately (buoyancy/ballast).
    """
    features = []
    for target in target_frames:
        try:
            trans = tf_buffer.lookup_transform(
                source_frame, target, time, rospy.Duration(0.1)
            )

            p = trans.transform.translation
            q = trans.transform.rotation

            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            # omit roll and pitch - only relative position and yaw to gate matters
            features.extend([p.x, p.y, p.z, yaw])

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            # return zeros
            features.extend([0.0] * 4)

    return features


def get_4axis_velocity(odom_velocity):
    """
    Extract 4-axis velocity from 6D odometry velocity.
    Input: [vx, vy, vz, wx, wy, wz]
    Output: [vx, vy, vz, wz]
    """
    return [odom_velocity[0], odom_velocity[1], odom_velocity[2], odom_velocity[5]]


def get_robot_state_at_time(tf_buffer, odom_data_cache, source_frame, time):
    """
    Get robot velocity and orientation at a specific time.
    TF for orientation and cached odometry for velocity.
    """
    # Get orientation from TF (always available and interpolated)
    try:
        transform = tf_buffer.lookup_transform(
            "odom", source_frame, time, rospy.Duration(0.1)
        )
        q = transform.transform.rotation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        roll, pitch, _ = euler_from_quaternion([q.x, q.y, q.z, q.w])
        orientation = [roll, pitch, yaw]
    except:
        orientation = [0.0, 0.0, 0.0]

    # Get velocity from cached odometry (find closest timestamp)
    if not odom_data_cache:
        return [0.0] * 6, orientation

    # Find odometry message closest to the requested time
    closest_odom = min(odom_data_cache, key=lambda x: abs((x[0] - time).to_sec()))
    velocity = closest_odom[1]

    return velocity, orientation


def process_bag(bag_path, target_frames, output_path):
    print(f"Processing bag: {bag_path}")
    bag = rosbag.Bag(bag_path)

    tf_buffer = tf2_ros.BufferCore(rospy.Duration(3600.0))

    data_inputs = []
    data_labels = []
    odom_data_cache = []

    print("Pass 1: Loading TF tree...")
    for topic, msg, t in bag.read_messages(topics=["/tf", "/tf_static"]):
        for transform in msg.transforms:
            tf_buffer.set_transform(transform, "default_authority")

    print("Pass 2: Extracting Action-State pairs...")

    count = 0

    # Now read action and odom
    for topic, msg, t in bag.read_messages(
        topics=["/taluy/cmd_vel", "/taluy/odometry"]
    ):

        if topic == "/taluy/odometry":
            # Use HEADER stamp, not bag time 't'
            stamp = msg.header.stamp
            v = msg.twist.twist.linear
            w = msg.twist.twist.angular
            velocity = [v.x, v.y, v.z, w.x, w.y, w.z]
            odom_data_cache.append((stamp, velocity))

            # Keep cache reasonably sized
            if len(odom_data_cache) > 200:
                odom_data_cache.pop(0)

        elif topic == "/taluy/cmd_vel":
            query_time = t

            try:
                # 1. Get Target Transforms
                target_features = []

                for target in target_frames:
                    # lookup_transform might fail if 't' is out of bounds of the TF buffer
                    trans = tf_buffer.lookup_transform_core(
                        "taluy/base_link", target, query_time
                    )
                    p = trans.transform.translation
                    
                    # Calculate bearing angle: the angle TO the target in body frame
                    # This is what the robot needs to turn to face the gate
                    # atan2(y, x) gives angle where 0 = ahead, positive = left
                    bearing_yaw = get_bearing_angle(p.x, p.y)
                    
                    # [x, y, z, bearing_yaw] (4D)
                    target_features.extend([p.x, p.y, p.z, bearing_yaw])

                # 2. Get Robot State (Vel + Orientation)
                # Query TF for Orientation
                trans_odom = tf_buffer.lookup_transform_core(
                    "odom", "taluy/base_link", query_time
                )
                q_o = trans_odom.transform.rotation
                roll, pitch, _ = euler_from_quaternion([q_o.x, q_o.y, q_o.z, q_o.w])
                # Only keep roll and pitch - yaw is already in relative target transform
                robot_orientation = [roll, pitch]

                # Query Cache for Velocity
                # Find closest odom message by time difference
                if not odom_data_cache:
                    continue  # Skip if no odom yet

                # Closest in time
                closest_odom = min(
                    odom_data_cache, key=lambda x: abs((x[0] - query_time).to_sec())
                )

                # If the gap is too large (e.g. > 0.2s), the data is stale. Skip.
                if abs((closest_odom[0] - query_time).to_sec()) > 0.2:
                    continue

                # Get 4-axis velocity [vx, vy, vz, wz] to match output space
                robot_vel = get_4axis_velocity(closest_odom[1])

                # Assemble
                # Input vector: [Relative Target (4*N), Ego Velocity (4), Ego Orientation (2)]
                # For single gate: 4 + 4 + 2 = 10D input
                input_vec = target_features + robot_vel + robot_orientation
                label_vec = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z]

                data_inputs.append(input_vec)
                data_labels.append(label_vec)

                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} valid samples...")

            except (
                tf2_ros.LookupException,
                tf2_ros.ExtrapolationException,
                tf2_ros.ConnectivityException,
            ):
                # If we fail to lookup, we DROP this sample.
                continue

    bag.close()

    np.savez(output_path, inputs=np.array(data_inputs), labels=np.array(data_labels))
    print(f"Saved {len(data_inputs)} valid samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training data from ROS bags.")
    parser.add_argument("bag_file", help="Path to input ROS bag")
    parser.add_argument(
        "--output", default="dataset.npz", help="Path to output NPZ file"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["gate_sawfish_link"],
        help="List of target TF frames",
    )

    args = parser.parse_args()

    process_bag(args.bag_file, args.targets, args.output)

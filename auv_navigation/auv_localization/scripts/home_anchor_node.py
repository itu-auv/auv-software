#!/usr/bin/env python3
"""
Anchors a permanent map->odom transform that:
 1. Starts as identity (map == odom at launch)
 2. Survives every odometry reset so map never moves
Provides a service to reset odometry via the EKF's set_pose API.
"""

import rospy
import numpy as np
import tf2_ros
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_from_matrix,
)
from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from std_srvs.srv import Empty, EmptyResponse
from robot_localization.srv import SetPose, SetPoseRequest


class MapOdomKeeper:
    def __init__(self):
        # Parameters
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "taluy/base_link")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.set_pose_service = rospy.get_param("~set_pose_service", "/taluy/set_pose")

        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Wait for EKF set_pose service
        rospy.loginfo(f"Waiting for EKF set_pose service '{self.set_pose_service}'...")
        rospy.wait_for_service(self.set_pose_service)
        self.set_pose_client = rospy.ServiceProxy(self.set_pose_service, SetPose)

        # Advertise our reset service
        rospy.Service("~reset_odom", Empty, self.handle_reset)

        # At startup, map == odom
        self.H_map_odom = np.eye(4)
        self._broadcast_transform()
        rospy.loginfo("map->odom initialized to identity")

    def _lookup_odom_to_base(self):
        """
        Lookup the transform from odom_frame to base_frame and return as a homogeneous matrix.
        Returns None if lookup fails.
        """
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, rospy.Time(0), rospy.Duration(1.0)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logerr(f"TF lookup failed: {e}")
            return None
        # Build homogeneous matrix
        trans = [
            tf_stamped.transform.translation.x,
            tf_stamped.transform.translation.y,
            tf_stamped.transform.translation.z,
        ]
        rot_q = [
            tf_stamped.transform.rotation.x,
            tf_stamped.transform.rotation.y,
            tf_stamped.transform.rotation.z,
            tf_stamped.transform.rotation.w,
        ]
        H = quaternion_matrix(rot_q)
        H[0:3, 3] = trans
        return H

    def _broadcast_transform(self):
        """
        Broadcast the current map->odom transform as a static transform.
        """
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame

        trans = translation_from_matrix(self.H_map_odom)
        rot = quaternion_from_matrix(self.H_map_odom)
        t.transform.translation.x = trans[0]
        t.transform.translation.y = trans[1]
        t.transform.translation.z = trans[2]
        t.transform.rotation.x = rot[0]
        t.transform.rotation.y = rot[1]
        t.transform.rotation.z = rot[2]
        t.transform.rotation.w = rot[3]

        self.broadcaster.sendTransform(t)

    def handle_reset(self, req):
        """
        Service callback to reset odometry.
        """
        rospy.loginfo(
            "'reset_odom' called: freezing map->odom and resetting EKF odometry."
        )

        # 1) Capture current map -> odom and odom -> base_link before reset

        H_map_odom_before = np.copy(self.H_map_odom)
        H_odom_base_before = self._lookup_odom_to_base()
        if H_odom_base_before is None:
            rospy.logwarn(
                "Could not lookup odom->base_link before reset. Aborting reset."
            )
            return EmptyResponse()

        rospy.loginfo(
            f"H_odom_base_before Z: {translation_from_matrix(H_odom_base_before)[2]}"
        )

        #! DEBUG
        # Print current transform for debugging
        H_map_base_before = H_map_odom_before.dot(H_odom_base_before)
        rospy.loginfo(f"Before reset: map>base translation: {H_map_base_before[0:3,3]}")

        # 2) Prepare and call the EKF set_pose service
        pose_with_cov = PoseWithCovarianceStamped()
        pose_with_cov.header.stamp = rospy.Time.now()
        pose_with_cov.header.frame_id = self.odom_frame
        pose_with_cov.pose.pose.orientation.w = 1.0
        pose_with_cov.pose.covariance = [0.0] * 36
        set_pose_request = SetPoseRequest(pose=pose_with_cov)

        try:
            self.set_pose_client(set_pose_request)
            self.H_map_odom = H_map_base_before
            self._broadcast_transform()
            rospy.loginfo("Broadcasted INTERMEDIATE map->odom")
        except rospy.ServiceException as e:
            rospy.logerr(f"set_pose service failed: {e}")
            return EmptyResponse()

        # Wait for H_odom_base_new to be stable before using
        rospy.sleep(1.5)

        # 3) Capture the stable odom -> base_link after the reset
        H_odom_base_new = self._lookup_odom_to_base()
        if H_odom_base_new is None:
            rospy.logwarn(
                "Could not lookup odom->base_link after reset. Aborting reset."
            )
            # What to do here?
            # Option A: Keep the intermediate H_map_odom (H_M_B_target). This assumes odom->base IS Identity,
            # which is unlikely after surfacing.
            # Option B: Revert to H_M_O_old. This would undo the reset from map's perspective.
            # Option C: Do nothing, H_map_odom is H_M_B_target. If H_O_B_new_surfaced is not Identity, robot will be offset.
            # For now, let's proceed but log a strong warning. The final transform will be based on M_B_target.
            # This means if H_O_B_new_surfaced lookup fails, the robot might be off by inv(Expected_H_O_B_new_surfaced)
            # A robust solution might try a few times or have a fallback.
            # Let's assume for now the intermediate broadcast is better than nothing if this fails.
            # The next line would use H_M_B_target.dot(np.linalg.inv(FAULTY_OR_MISSING_H_O_B_new_surfaced))
            # which could be bad.
            # Let's just return, leaving H_map_odom as H_M_B_target. This means we *assume* odom->base_link became Identity
            # and stayed that way. This is likely wrong but might be less jarring than other options if H_O_B_new fails.
            return EmptyResponse()
        rospy.loginfo(
            f"H_odom_base_new Z: {translation_from_matrix(H_odom_base_new)[2]}"
        )

        # 4) Compute the new map -> odom transform
        # H_map_odom_new = H_map_odom_old * H_odom_base_old * inv(H_odom_base_new)
        H_map_odom_new = H_map_odom_before.dot(H_odom_base_before).dot(
            np.linalg.inv(H_odom_base_new)
        )
        # Force the z translation to remain the same as before
        H_map_odom_new[2, 3] = H_map_odom_before[2, 3]

        self.H_map_odom = H_map_odom_new
        self._broadcast_transform()
        rospy.loginfo("map->odom preserved after reset_odom.")

        H_map_base_new = H_map_odom_new.dot(H_odom_base_new)
        rospy.loginfo(f"After reset: map>base translation: {H_map_base_new[0:3,3]}")

        return EmptyResponse()


def main():
    rospy.init_node("home_anchor_node")
    keeper = MapOdomKeeper()
    rospy.spin()


if __name__ == "__main__":
    main()

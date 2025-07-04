#!/usr/bin/env python3
"""
pointcloud_clusterer.py

• /slalom_red_cloud  ve /slalom_white_cloud topic’lerinden PointCloud2 alır  
• Optik frame’de y-ekseni dikkate alınmadan (yalnız x–z düzleminde) DBSCAN kümeleme yapar  
• Her küme için TF child-frame yayınlar:   red_pipe_cluster_N  /  white_pipe_cluster_N
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from dynamic_reconfigure.server import Server
from auv_mapping.cfg import SlalomClusteringConfig


class PointCloudClusterer:
    def __init__(self):
        rospy.init_node("pointcloud_clusterer")

        # ── parametreler ────────────────────────────────────────────────────
        red_topic   = rospy.get_param("~red_cloud_topic",   "/slalom_red_cloud")
        white_topic = rospy.get_param("~white_cloud_topic", "/slalom_white_cloud")
        self.parent_frame = rospy.get_param(
            "~parent_frame", "taluy/camera_depth_optical_frame"
        )

        # kümeleme ayarları
        self.max_distance = rospy.get_param("~max_distance", 30.0)
        self.eps = rospy.get_param("~eps", 0.10)
        self.min_samples = rospy.get_param("~min_samples", 10)
        self.downsample_n = rospy.get_param("~downsample_n", 1)

        # ── dynamic reconfigure ─────────────────────────────────────────────
        self.srv = Server(SlalomClusteringConfig, self.reconfigure_cb)

        # ── aboneler & TF yayıncısı ────────────────────────────────────────
        self.tf_br = TransformBroadcaster()

        self.red_sub = rospy.Subscriber(
            red_topic, PointCloud2, self.pointcloud_callback, callback_args="red"
        )
        self.white_sub = rospy.Subscriber(
            white_topic, PointCloud2, self.pointcloud_callback, callback_args="white"
        )

        rospy.loginfo("PointCloud clusterer with TF broadcasting initialized")

    # ═══════════════════════════════════════════════════════════════════════
    def reconfigure_cb(self, config, level):
        """Dynamic reconfigure callback"""
        rospy.loginfo("Reconfiguring clustering parameters")
        self.max_distance = config.max_distance
        self.eps = config.eps
        self.min_samples = config.min_samples
        self.downsample_n = config.downsample_n
        return config

    def pointcloud_callback(self, msg: PointCloud2, color: str):
        """
        Tek callback; color argümanı 'red' veya 'white'
        """
        print("emin selam ya")
        # 1) Noktaları oku (downsample)
        pts_full = [
            (p[0], p[1], p[2])
            for i, p in enumerate(
                pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            )
            if i % self.downsample_n == 0
        ]
        if not pts_full:
            return

        pts = np.asarray(pts_full, dtype=np.float32)

        # 2) max uzaklık filtresi (3-B norm)
        dmask = np.linalg.norm(pts, axis=1) < self.max_distance
        pts = pts[dmask]
        if pts.shape[0] < self.min_samples:
            return

        # 3) x-z düzlemine indirgeme  → 2-B matrisi
        xz = pts[:, [0, 2]]  # shape (N,2)

        # 4) DBSCAN (yalnız x–z)
        labels = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",       # default
            algorithm="ball_tree",
        ).fit_predict(xz)

        # 5) Küme yayınla
        for lbl in set(labels):
            if lbl == -1:
                continue  # outlier

            mask = labels == lbl
            cluster_pts = pts[mask]         # hâlâ 3-B
            centroid = cluster_pts.mean(axis=0)  # (x̄,ȳ,z̄)

            tfm = TransformStamped()
            tfm.header.stamp    = msg.header.stamp
            tfm.header.frame_id = self.parent_frame
            tfm.child_frame_id  = f"{color}_pipe_cluster_{lbl}"

            tfm.transform.translation.x = float(centroid[0])
            tfm.transform.translation.y = float(centroid[1])  # y ortalaması
            tfm.transform.translation.z = float(centroid[2])
            tfm.transform.rotation.w    = 1.0  # (0,0,0,1) – identite

            self.tf_br.sendTransform(tfm)
            print(
                f"Published TF: {tfm.child_frame_id} at "
                f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
            )
            rospy.logdebug(
                f"[{tfm.child_frame_id}] → ({centroid[0]:.2f}, "
                f"{centroid[1]:.2f}, {centroid[2]:.2f})"
            )

# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        PointCloudClusterer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

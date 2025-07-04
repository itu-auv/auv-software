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
from geometry_msgs.msg import TransformStamped, PointStamped
import tf2_geometry_msgs
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from dynamic_reconfigure.server import Server
from auv_mapping.cfg import SlalomClusteringConfig
from auv_msgs.msg import Pipe, Pipes


class PointCloudClusterer:
    def __init__(self):
        rospy.init_node("pointcloud_clusterer")

        # ── parametreler ────────────────────────────────────────────────────
        red_topic   = rospy.get_param("~red_cloud_topic",   "/slalom_red_cloud")
        white_topic = rospy.get_param("~white_cloud_topic", "/slalom_white_cloud")
        self.parent_frame = rospy.get_param(
            "~parent_frame", "taluy/camera_depth_optical_frame"
        )
        self.target_frame = rospy.get_param("~target_frame", "odom")

        # kümeleme ayarları
        self.max_distance = rospy.get_param("~max_distance", 30.0)
        self.eps = rospy.get_param("~eps", 0.10)
        self.min_samples = rospy.get_param("~min_samples", 10)
        self.downsample_n = rospy.get_param("~downsample_n", 1)

        # ── dynamic reconfigure ─────────────────────────────────────────────
        self.srv = Server(SlalomClusteringConfig, self.reconfigure_cb)

        # ── aboneler & TF yayıncısı ────────────────────────────────────────
        self.tf_br = TransformBroadcaster()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.pipes_pub = rospy.Publisher("/taluy/slalom_pipes", Pipes, queue_size=10)

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

        # 5) Küme ve P boru mesajlarını yayınla
        pipes_msg = Pipes()
        pipes_msg.header.stamp = msg.header.stamp
        pipes_msg.header.frame_id = self.target_frame

        for lbl in set(labels):
            if lbl == -1:
                continue  # outlier

            mask = labels == lbl
            cluster_pts = pts[mask]         # hâlâ 3-B
            centroid = cluster_pts.mean(axis=0)  # (x̄,ȳ,z̄)

            # a) TF yayınla (orijinal işlevsellik)
            tfm = TransformStamped()
            tfm.header.stamp    = msg.header.stamp
            tfm.header.frame_id = self.parent_frame
            tfm.child_frame_id  = f"{color}_pipe_cluster_{lbl}"
            tfm.transform.translation.x = float(centroid[0])
            tfm.transform.translation.y = float(centroid[1])
            tfm.transform.translation.z = float(centroid[2])
            tfm.transform.rotation.w    = 1.0
            self.tf_br.sendTransform(tfm)
            print(
                f"Published TF: {tfm.child_frame_id} at "
                f"({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})"
            )
            rospy.logdebug(
                f"[{tfm.child_frame_id}] → ({centroid[0]:.2f}, "
                f"{centroid[1]:.2f}, {centroid[2]:.2f})"
            )

            # b) odom'a dönüştür ve Pipe mesajı oluştur
            try:
                source_point = PointStamped()
                source_point.header.frame_id = self.parent_frame
                source_point.header.stamp = msg.header.stamp
                source_point.point.x = float(centroid[0])
                source_point.point.y = float(centroid[1])
                source_point.point.z = float(centroid[2])

                # Daha sağlam bir dönüşüm için lookup_transform ve do_transform_point kullan
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.parent_frame,
                    msg.header.stamp,
                    rospy.Duration(1.0)
                )
                transformed_point = tf2_geometry_msgs.do_transform_point(source_point, transform)

                pipe = Pipe()
                pipe.color = color
                pipe.position = transformed_point.point
                pipes_msg.pipes.append(pipe)

            except Exception as e:
                rospy.logwarn(f"Could not transform point for {color}_pipe_cluster_{lbl}: {e}")

        # c) Pipes mesajını yayınla
        if pipes_msg.pipes:
            self.pipes_pub.publish(pipes_msg)
            rospy.loginfo(f"Published {len(pipes_msg.pipes)} {color} pipes to /slalom_pipes")

# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        PointCloudClusterer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

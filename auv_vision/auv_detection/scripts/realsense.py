#!/usr/bin/env python3
#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf_trans

class RealsensePointCloudCorrection:
    def __init__(self):
        self.depthsub = rospy.Subscriber("/taluy/camera/depth/points", PointCloud2, self.pointcloudcallback)
        self.corrected_pcd_pub = rospy.Publisher("/corrected_point_cloud", PointCloud2, queue_size=10)
        self.downsample_pub = rospy.Publisher("/downsampled_point_cloud", PointCloud2, queue_size=10)
    def correct_refraction(self,raw_point_cloud, n, d_h, t_h, eta_water, eta_glass):
        """
        Su altı kırılma etkilerini düzelten fonksiyon.

        :param raw_point_cloud: Nx3 numpy array (hatalı 3B noktalar).
        :param n: Arayüz normal vektörü (örneğin [0, 0, 1]).
        :param d_h, t_h: Kalibrasyon parametreleri.
        :param eta_water, eta_glass: Kırılma indisleri.
        :return: Düzeltilmiş Nx3 point cloud.
        """
        n = np.array(n) / np.linalg.norm(n)  # Normalizasyon
        corrected_points = []

        for point in raw_point_cloud:
            # 1. Işını suda geriye doğru izle (point -> muhafaza)
            X_w = point / np.linalg.norm(point)  # Su içindeki yön

            # 2. Muhafaza-su arayüzünde kırılma (Snell yasası)
            theta_w = np.arccos(np.dot(-X_w, n))
            sin_theta_h = (eta_water / eta_glass) * np.sin(theta_w)
            theta_h = np.arcsin(sin_theta_h)

            # 3. Muhafaza içindeki yönü hesapla
            rotation_axis = np.cross(n, -X_w) / np.linalg.norm(np.cross(n, -X_w))
            X_h = self.rotate_vector(-X_w, rotation_axis, theta_h - theta_w)

            # 4. Muhafaza-hava arayüzüne olan mesafeyi bul
            x_o = point - (t_h / np.cos(theta_h)) * X_h  # Muhafaza çıkış noktası

            # 5. Hava içindeki yönü hesapla
            sin_theta_a = (eta_glass / 1.0) * np.sin(theta_h)
            theta_a = np.arcsin(sin_theta_a)
            X_a = self.rotate_vector(X_h, rotation_axis, theta_a - theta_h)

            # 6. Pinhole'a olan gerçek mesafe
            lambda_a = d_h / np.cos(theta_a)
            real_point = x_o - lambda_a * X_a
            corrected_points.append(real_point)

        return np.array(corrected_points)

    def rotate_vector(self,v, axis, angle):
        """ Rodrigues dönüş formülü """
        return (v * np.cos(angle)+
                np.cross(axis, v) * np.sin(angle) +
                axis * np.dot(axis, v) * (1 - np.cos(angle)))

    def downsample(self, point_cloud, voxel_size):
        """
        Nokta bulutunu downsample eden fonksiyon.

        :param point_cloud: Nx3 numpy array (3B noktalar).
        :param voxel_size: Voxel boyutu.
        :return: Downsample edilmiş Nx3 numpy array.
        """
        open3d_pcd = o3d.geometry.PointCloud()
        open3d_pcd.points = o3d.utility.Vector3dVector(point_cloud)
        downsampled_pcd = open3d_pcd.voxel_down_sample(voxel_size)
        return np.asarray(downsampled_pcd.points)

    def pointcloudcallback(self,msg):
        # 1. PointCloud2 mesajından noktaları çıkar
        raw_point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        raw_point_cloud_list = list(raw_point_cloud) # Convert generator to list
        raw_point_cloud_np = np.array(raw_point_cloud_list, dtype=np.float32) # Convert list to numpy array

        # Downsample uygula
        downsampled_point_cloud = self.downsample(raw_point_cloud_np, 0.1)
        self.downsample_pub.publish(pc2.create_cloud_xyz32(msg.header, downsampled_point_cloud))
        # 2. Kırılma düzeltmesi uygula
        corrected_points = self.correct_refraction(
            downsampled_point_cloud,  # Downsample edilmiş nokta bulutunu kullan
            n=[0, 0, 1],          # Kamera muhafazası normali (z yönünde)
            d_h=10.0,             # Pinhole-muhafaza mesafesi (mm)
            t_h=5.0,              # Muhafaza kalınlığı (mm)
            eta_water=1.33,       # Suyun kırılma indisi
            eta_glass=1.49        # Akrilik kırılma indisi
        )
        self.corrected_pcd_pub.publish(pc2.create_cloud_xyz32(msg.header, corrected_points))


if __name__ == '__main__':
    rospy.init_node('depth_image_processor')
    depth_processor = RealsensePointCloudCorrection()
    rospy.spin()

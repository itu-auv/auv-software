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
        rospy.loginfo("Realsense Point Cloud Correction Node Initialized")
        self.depthsub = rospy.Subscriber("/taluy/camera/depth/points", PointCloud2, self.pointcloudcallback)
        self.corrected_pcd_pub = rospy.Publisher("/corrected_point_cloud", PointCloud2, queue_size=10)
        self.downsample_pub = rospy.Publisher("/downsampled_point_cloud", PointCloud2, queue_size=10)
        self.voxel_size = 0.01  # Voxel boyutu
    def correct_refraction(self, point_cloud, eta_water):
        corrected_points = []
        eta = eta_water  # Hava -> cam -> su geçişi varsayımıyla cam su arasına odaklanıyoruz
        for pt in point_cloud:
            x, y, z = pt

            # Nokta vektörü ile Z ekseni arasındaki açıyı hesapla
            r = np.sqrt(x**2 + z**2)
            theta = np.arccos(z / r)
            angle_deg=np.degrees(theta)
            error_percent = -25 - 0.03 * (angle_deg**2)

            # Convert percentage error to correction factor
            # If error is -25%, then correction factor should be 1/(1 + (-25/100)) = 1/0.75 = 1.33
            correction_factor = 1 / (1 + (error_percent/100))            # Z düzelt
            corrected_z = z * correction_factor
            # Ölçeklemenin yalnızca z'ye değil, tüm vektöre uygulanması daha doğru olur
            scale = corrected_z / z
            corrected_point = np.array([x , y , corrected_z])
            corrected_points.append(corrected_point)
        return np.array(corrected_points, dtype=np.float32)

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
        downsampled_point_cloud = self.downsample(raw_point_cloud_np, self.voxel_size)  # Voxel boyutunu ayarla
        self.downsample_pub.publish(pc2.create_cloud_xyz32(msg.header, downsampled_point_cloud))
        # 2. Kırılma düzeltmesi uygula
        corrected_points = self.correct_refraction(
            downsampled_point_cloud,  # Downsample edilmiş nokta bulutunu kullan
            eta_water=1.33,       # Suyun kırılma indisi
        )
        self.corrected_pcd_pub.publish(pc2.create_cloud_xyz32(msg.header, corrected_points))


if __name__ == '__main__':
    rospy.init_node('depth_image_processor')
    depth_processor = RealsensePointCloudCorrection()
    rospy.spin()

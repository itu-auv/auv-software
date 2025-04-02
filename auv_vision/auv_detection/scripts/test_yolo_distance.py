#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point
import tf.transformations as tf_trans

class RealsensePointCloudCorrection:
    def __init__(self):
        # Camera parameters (calibrate these for your setup)
        self.baseline = 70.0    # mm (D435 baseline)
        self.d_h = 15.0         # mm (pinhole to housing)
        self.t_h = 7.5          # mm (housing thickness)
        self.eta_air = 1.0
        self.eta_glass = 1.49   # Acrylic
        self.eta_water = 1.33
        self.interface_normal = np.array([0, 0, 1])  # Housing normal
        
        self.depthsub = rospy.Subscriber("/taluy/camera/depth/points", PointCloud2, self.pointcloudcallback)
        self.corrected_pcd_pub = rospy.Publisher("/corrected_point_cloud", PointCloud2, queue_size=10)
        self.downsample_pub = rospy.Publisher("/downsampled_point_cloud", PointCloud2, queue_size=10)

    def correct_refraction(self, raw_point_cloud):
        """Dual-ray structured light refraction correction"""
        corrected_points = []
        
        for point in raw_point_cloud:
            # Convert to millimeters for calculations
            point_mm = point * 1000  # ROS uses meters, convert to mm
            
            # Receiver path (sensor to object)
            rec_origin = np.array([self.baseline/2, 0, 0])
            rec_dir = (point_mm - rec_origin) / np.linalg.norm(point_mm - rec_origin)
            rec_exit = self.trace_ray(rec_origin, rec_dir)
            
            # Projector path (projector to object)
            proj_origin = np.array([-self.baseline/2, 0, 0])
            proj_dir = (point_mm - proj_origin) / np.linalg.norm(point_mm - proj_origin)
            proj_exit = self.trace_ray(proj_origin, proj_dir)
            
            # Find intersection in water
            intersection = self.ray_intersection(
                (proj_exit, proj_dir),
                (rec_exit, rec_dir)
            )
            
            if intersection is not None:
                corrected_points.append(intersection / 1000)  # Convert back to meters
        
        return np.array(corrected_points)

    def trace_ray(self, origin, direction):
        """Trace ray through air->housing->water interfaces"""
        # First interface (air to housing)
        theta_air = np.arccos(np.clip(np.dot(direction, self.interface_normal), -1, 1))
        sin_theta_housing = (self.eta_air / self.eta_glass) * np.sin(theta_air)
        
        # Handle total internal reflection
        if np.abs(sin_theta_housing) > 1.0:
            return None
        
        theta_housing = np.arcsin(sin_theta_housing)
        dir_housing = self.refract_vector(direction, self.interface_normal, self.eta_air, self.eta_glass)
        
        # Second interface (housing to water)
        sin_theta_water = (self.eta_glass / self.eta_water) * np.sin(theta_housing)
        if np.abs(sin_theta_water) > 1.0:
            return None
        
        theta_water = np.arcsin(sin_theta_water)
        dir_water = self.refract_vector(dir_housing, self.interface_normal, self.eta_glass, self.eta_water)
        
        # Calculate path lengths
        path_air = self.d_h / np.cos(theta_air)
        path_housing = self.t_h / np.cos(theta_housing)
        
        return origin + path_air * direction + path_housing * dir_housing

    def refract_vector(self, incident, normal, eta1, eta2):
        """Numerically stable Snell's law implementation"""
        incident = incident / np.linalg.norm(incident)
        cos_theta1 = np.dot(normal, incident)
        
        # Handle potential numerical issues
        ratio = eta1 / eta2
        radicand = 1 - (ratio**2 * (1 - cos_theta1**2))
        if radicand < 0:  # Total internal reflection
            return None
        
        cos_theta2 = np.sqrt(radicand)
        return ratio * incident + (ratio * cos_theta1 - cos_theta2) * normal

    def ray_intersection(self, ray1, ray2, epsilon=1e-6):
        """3D ray intersection with enhanced stability"""
        # Ray 1: P + t*u
        # Ray 2: Q + s*v
        P = ray1[0]
        u = ray1[1]
        Q = ray2[0]
        v = ray2[1]
        
        w0 = P - Q
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)
        
        denominator = a*c - b*b
        if np.abs(denominator) < epsilon:
            return None  # Parallel rays
        
        sc = (b*e - c*d) / denominator
        tc = (a*e - b*d) / denominator
        
        # Return midpoint between closest points
        point1 = P + sc * u
        point2 = Q + tc * v
        return (point1 + point2) / 2

    def downsample(self, point_cloud, voxel_size):
        """Open3D-based downsampling"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        return np.asarray(pcd.voxel_down_sample(voxel_size).points)

    def pointcloudcallback(self, msg):
        # Convert ROS message to numpy array
        raw_points = np.array((list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        
        # Downsample first
        downsampled = self.downsample(raw_points, 0.1)
        self.downsample_pub.publish(pc2.create_cloud_xyz32(msg.header, downsampled))
        
        # Apply refraction correction
        corrected = self.correct_refraction(downsampled)
        
        if len(corrected) > 0:
            self.corrected_pcd_pub.publish(pc2.create_cloud_xyz32(msg.header, corrected))

if __name__ == '__main__':
    rospy.init_node('depth_image_processor')
    processor = RealsensePointCloudCorrection()
    rospy.spin()
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
        # Camera parameters (from PDF and D435 specs)
        self.baseline = 70.0  # mm (distance between projector/receiver)
        self.d_h = 10.0       # mm (pinhole to housing interface)
        self.t_h = 5.0        # mm (housing thickness)
        self.eta_air = 1.0
        self.eta_glass = 1.49  # Acrylic refractive index
        self.eta_water = 1.33  # Water refractive index
        
        self.depthsub = rospy.Subscriber("/taluy/camera/depth/points", PointCloud2, self.pointcloudcallback)
        self.corrected_pcd_pub = rospy.Publisher("/corrected_point_cloud", PointCloud2, queue_size=10)
        self.downsample_pub = rospy.Publisher("/downsampled_point_cloud", PointCloud2, queue_size=10)

    def correct_refraction(self, raw_point_cloud):
        """
        Structured light refraction correction based on PDF model
        Implements dual-ray tracing for both projector and receiver
        """
        corrected_points = []
        interface_normal = np.array([0, 0, 1])  # Assuming flat housing interface
        
        for point in raw_point_cloud:
            # Convert to meters to millimeters (assuming input is in meters)
            point_mm = point * 1000
            
            # Receiver ray tracing (from measured point to sensor)
            rec_origin = np.array([self.baseline/2, 0, 0])  # Receiver position
            rec_dir = (rec_origin - point_mm) / np.linalg.norm(rec_origin - point_mm)
            
            # Trace receiver ray through interfaces
            rec_water_pt = self.trace_ray(rec_origin, rec_dir, interface_normal)
            
            # Projector ray tracing (from projector to object)
            proj_origin = np.array([-self.baseline/2, 0, 0])  # Projector position
            proj_dir = (point_mm - proj_origin) / np.linalg.norm(point_mm - proj_origin)
            
            # Trace projector ray through interfaces
            proj_water_pt = self.trace_ray(proj_origin, proj_dir, interface_normal)
            
            # Find intersection point (true 3D location)
            intersection = self.ray_intersection(
                (proj_water_pt, proj_dir),
                (rec_water_pt, -rec_dir)
            )
            
            if intersection is not None:
                corrected_points.append(intersection / 1000)  # Convert back to meters
        
        return np.array(corrected_points)

    def trace_ray(self, origin, direction, normal):
        """Trace a ray through both interfaces (air->housing->water)"""
        # First refraction (air to housing)
        theta_air = np.arccos(np.dot(direction, normal))
        theta_housing = np.arcsin((self.eta_air/self.eta_glass) * np.sin(theta_air))
        dir_housing = self.refract_vector(direction, normal, self.eta_air, self.eta_glass)
        
        # Second refraction (housing to water)
        theta_water = np.arcsin((self.eta_glass/self.eta_water) * np.sin(theta_housing))
        dir_water = self.refract_vector(dir_housing, normal, self.eta_glass, self.eta_water)
        
        # Calculate total path length
        path_air = self.d_h / np.cos(theta_air)
        path_housing = self.t_h / np.cos(theta_housing)
        
        return origin + path_air * direction + path_housing * dir_housing

    def refract_vector(self, incident, normal, eta1, eta2):
        """Vector form of Snell's law"""
        cos_theta1 = np.dot(normal, incident)
        ratio = eta1 / eta2
        cos_theta2 = np.sqrt(1 - (ratio**2 * (1 - cos_theta1**2)))
        
        return ratio * incident + (ratio * cos_theta1 - cos_theta2) * normal

    def ray_intersection(self, ray1, ray2, epsilon=1e-6):
        """Find closest point between two 3D rays"""
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
        
        denom = a*c - b*b
        if abs(denom) < epsilon:
            return None
            
        sc = (b*e - c*d) / denom
        tc = (a*e - b*d) / denom
        
        return P + sc*u  # Return midpoint between closest points

    def downsample(self, point_cloud, voxel_size):
        """Downsample point cloud using Open3D"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        down_pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(down_pcd.points)

    def pointcloudcallback(self, msg):
        # Convert ROS PointCloud2 to numpy array
        raw_points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        
        # Downsample first
        downsampled_points = self.downsample(raw_points, 0.1)
        self.downsample_pub.publish(pc2.create_cloud_xyz32(msg.header, downsampled_points))
        
        # Apply refraction correction
        corrected_points = self.correct_refraction(downsampled_points)
        
        # Publish corrected cloud
        if len(corrected_points) > 0:
            self.corrected_pcd_pub.publish(pc2.create_cloud_xyz32(msg.header, corrected_points))

if __name__ == '__main__':
    rospy.init_node('depth_image_processor')
    depth_processor = RealsensePointCloudCorrection()
    rospy.spin()
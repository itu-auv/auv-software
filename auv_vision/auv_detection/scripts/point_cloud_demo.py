#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import copy

def create_flat_surface():
    # Create a flat rectangular surface
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Add some random noise
    Z += np.random.normal(0, 0.01, Z.shape)
    
    # Combine into points
    points = np.zeros((X.size, 3))
    points[:, 0] = X.flatten()
    points[:, 1] = Y.flatten()
    points[:, 2] = Z.flatten()
    
    # Rotation angles in radians
    theta_x = np.radians(20)  # 20 derece x etrafında
    theta_y = np.radians(30)  # 30 derece y etrafında
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    # Combined rotation matrix
    R = Ry @ Rx
    
    # Apply rotations
    points = points @ R.T
    
    # Lift the surface up
    points[:, 2] += 0.5
    
    return points, R

def calculate_euler_angles(normals):
    """
    Calculate Euler angles from surface normals relative to z-axis basis vector
    Returns: roll, pitch, yaw in degrees
    """
    # Calculate mean normal vector
    mean_normal = np.mean(normals, axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)  # Normalize
    
    # Basis vector (z-axis)
    basis = np.array([0, 0, 1])
    
    # Calculate angles
    # Pitch (rotation around y-axis)
    pitch = np.arctan2(mean_normal[0], mean_normal[2])
    
    # Roll (rotation around x-axis)
    roll = -np.arctan2(mean_normal[1], mean_normal[2])
    
    # Yaw (rotation around z-axis)
    # For a surface normal, yaw can be calculated from the x,y components
    yaw = np.arctan2(mean_normal[1], mean_normal[0])
    
    # Convert to degrees
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    return roll_deg, pitch_deg, yaw_deg, mean_normal

def main():
    print("Creating flat surface point cloud...")
    xyz, rotation_matrix = create_flat_surface()
    
    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # Calculate Euler angles
    normals_np = np.asarray(pcd.normals)
    roll, pitch, yaw, mean_normal = calculate_euler_angles(normals_np)
    print("\nSurface Orientation:")
    print(f"Mean Normal Vector: [{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}]")
    print(f"Roll  (x-rotation): {roll:.2f} degrees")
    print(f"Pitch (y-rotation): {pitch:.2f} degrees")
    print(f"Yaw   (z-rotation): {yaw:.2f} degrees")
    
    # Create world coordinate frame
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    
    # Create surface coordinate frame
    surface_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    
    # Calculate surface center
    center = np.mean(xyz, axis=0)
    
    # Transform surface frame
    # First rotate
    R = o3d.geometry.get_rotation_matrix_from_xyz([np.radians(20), np.radians(30), 0])
    surface_frame.rotate(R, center=[0, 0, 0])
    # Then translate to surface center
    surface_frame.translate(center)
    
    # Paint the point cloud
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color
    
    # Single visualization with normal vectors
    print("\nVisualizing point cloud with normal vectors...")
    o3d.visualization.draw_geometries([pcd, world_frame, surface_frame],
                                    zoom=0.5,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[0.0, 0.0, 0.0],
                                    up=[-0.0694, -0.9768, 0.2024],
                                    point_show_normal=True)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import copy

def create_flat_surface(theta_x=20, theta_y=30, theta_z=0):
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
    theta_x = np.radians(theta_x)
    theta_y = np.radians(theta_y)
    theta_z = np.radians(theta_z)
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
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
    # Get rotation angles from user
    print("Enter rotation angles in degrees:")
    theta_x = float(input("X rotation angle: "))
    theta_y = float(input("Y rotation angle: "))
    theta_z = float(input("Z rotation angle: "))
    
    print("\nCreating flat surface point cloud...")
    xyz, input_rotation_matrix = create_flat_surface(theta_x, theta_y, theta_z)
    
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
    
    # Calculate rotation matrix from estimated angles
    estimated_R = o3d.geometry.get_rotation_matrix_from_xyz([np.radians(roll), np.radians(pitch), np.radians(yaw)])
    
    print("\nInput Angles:")
    print(f"X rotation: {theta_x:.2f}°")
    print(f"Y rotation: {theta_y:.2f}°")
    print(f"Z rotation: {theta_z:.2f}°")
    print("\nInput Rotation Matrix:")
    print(np.array2string(input_rotation_matrix, precision=3, suppress_small=True))
    
    print("\nEstimated Surface Orientation:")
    print(f"Mean Normal Vector: [{mean_normal[0]:.3f}, {mean_normal[1]:.3f}, {mean_normal[2]:.3f}]")
    print(f"Roll  (x-rotation): {roll:.2f}°")
    print(f"Pitch (y-rotation): {pitch:.2f}°")
    print(f"Yaw   (z-rotation): {yaw:.2f}°")
    print("\nEstimated Rotation Matrix:")
    print(np.array2string(estimated_R, precision=3, suppress_small=True))
    
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
    
    # Transform surface frame using input angles
    R = o3d.geometry.get_rotation_matrix_from_xyz([np.radians(theta_x), np.radians(theta_y), np.radians(theta_z)])
    surface_frame.rotate(R, center=[0, 0, 0])
    surface_frame.translate(center)
    
    # Paint the point cloud
    pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray color
    
    # Visualize point cloud with normal vectors and coordinate frames
    print("\nVisualizing point cloud with normal vectors and coordinate frames...")
    print("Red axis: X, Green axis: Y, Blue axis: Z")
    o3d.visualization.draw_geometries([pcd, world_frame, surface_frame],
                                    zoom=0.5,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[0.0, 0.0, 0.0],
                                    up=[-0.0694, -0.9768, 0.2024],
                                    point_show_normal=True)

if __name__ == "__main__":
    main()
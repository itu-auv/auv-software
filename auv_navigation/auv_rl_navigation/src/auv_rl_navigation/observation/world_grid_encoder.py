#!/usr/bin/env python3
"""
World Grid Encoder - Convert TF transforms into a robot-based 3D voxel grid representation
"""

import rospy
import tf2_ros
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class WorldGridEncoder:
    """Creates a vehicle-centric 3D voxel grid from TF map."""

    def __init__(
        self,
        grid_dim_xy=10,
        grid_dim_z=2,
        cell_size_xy=0.7,
        cell_size_z=1.0,
        object_frames=None,
        visualize=True,
    ):
        """
        Initialize the World Grid Encoder.

        Args:
            grid_dim_xy (int): Horizontal (x-y) grid dimension (10x10 grid)
            grid_dim_z (int): Vertical (z) grid dimension / depth layers (2 layers)
            cell_size_xy (float): Cell size in X-Y plane (0.7m)
            cell_size_z (float): Layer height in Z axis (1.0m)
            object_frames (list): List of object frame names to track
            visualize (bool): Enable visualization markers
        """
        self.grid_dim_xy = grid_dim_xy
        self.grid_dim_z = grid_dim_z
        self.cell_size_xy = cell_size_xy
        self.cell_size_z = cell_size_z

        # Grid coverage area
        self.grid_range_xy = (grid_dim_xy * cell_size_xy) / 2.0  # ±3.5m
        self.grid_range_z = grid_dim_z * cell_size_z  # 2.0m total

        self.object_frames = object_frames if object_frames else []
        self.num_classes = len(self.object_frames)  # channels in grid

        self.visualize = visualize

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Visualization publishers
        if self.visualize:
            self.grid_pub = rospy.Publisher(
                "/auv_rl/grid_visualization", MarkerArray, queue_size=1
            )
            self.object_pub = rospy.Publisher(
                "/auv_rl/object_markers", MarkerArray, queue_size=1
            )

            # Define colors for different object classes
            self.class_colors = self._generate_class_colors()

        rospy.loginfo(
            f"WorldGridEncoder initialized: {grid_dim_xy}x{grid_dim_xy}x{grid_dim_z} grid, "
            f"cell_size={cell_size_xy}m, coverage=±{self.grid_range_xy}m, "
            f"tracking {self.num_classes} object types, visualization={'ON' if visualize else 'OFF'}"
        )

    def create_grid(self, base_frame="taluy/base_link"):
        """
        Create a vehicle-centric 3D voxel grid.

        Args:
            base_frame (str): The base frame for the vehicle (default: taluy/base_link)

        Returns:
            np.ndarray: Grid of shape (grid_dim_xy, grid_dim_xy, grid_dim_z, num_classes)
                       [y_index, x_index, z_index, object_class]
        """
        grid = np.zeros(
            (self.grid_dim_xy, self.grid_dim_xy, self.grid_dim_z, self.num_classes),
            dtype=np.float32,
        )

        object_positions = []  # For visualization

        for obj_idx, obj_frame in enumerate(self.object_frames):
            try:
                # Lookup transform from base_link to object frame
                transform = self.tf_buffer.lookup_transform(
                    base_frame, obj_frame, rospy.Time(0), rospy.Duration(0.1)
                )

                # Extract position in vehicle frame
                x = transform.transform.translation.x  # Forward/backward
                y = transform.transform.translation.y  # Left/right
                z = transform.transform.translation.z  # Up/down

                # Convert world coordinates to grid indices
                grid_x, grid_y, grid_z = self._world_to_grid(x, y, z)

                if self._in_bounds(grid_x, grid_y, grid_z):
                    # Weight by 3D distance (closer objects have higher values)
                    distance_3d = np.sqrt(x**2 + y**2 + z**2)
                    max_distance = np.sqrt(
                        (self.grid_range_xy * 1.5) ** 2 + self.grid_range_z**2
                    )
                    weight = max(0.0, 1.0 - distance_3d / max_distance)

                    # Assign weighted value to grid cell
                    grid[grid_y, grid_x, grid_z, obj_idx] = weight

                    # Store for visualization
                    object_positions.append(
                        {
                            "x": x,
                            "y": y,
                            "z": z,
                            "grid_x": grid_x,
                            "grid_y": grid_y,
                            "grid_z": grid_z,
                            "class_idx": obj_idx,
                            "weight": weight,
                            "frame": obj_frame,
                        }
                    )

            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                # Object not visible or TF not available
                continue

        # Visualize if enabled
        if self.visualize:
            self._visualize_grid(grid, base_frame)
            self._visualize_objects(object_positions, base_frame)

        return grid

    def _world_to_grid(self, x, y, z):
        """
        Convert world coordinates to grid indices.

        Coordinate system (base_link):
        - X: Forward (+) / Backward (-)
        - Y: Left (+) / Right (-)
        - Z: Up (+) / Down (-)

        Grid indices:
        - grid_x: 0 (left) → 9 (right)
        - grid_y: 0 (far/forward) → 9 (near/backward)
        - grid_z: 0 (down) → 1 (up)

        Args:
            x, y, z (float): Position in base_link frame (meters)

        Returns:
            tuple: (grid_x, grid_y, grid_z) indices
        """
        # X axis: forward is positive, grid goes top to bottom (0: far, 9: near)
        grid_y = int((self.grid_range_xy - x) / self.cell_size_xy)

        # Y axis: left is positive, grid goes left to right (0: left, 9: right)
        grid_x = int((y + self.grid_range_xy) / self.cell_size_xy)

        # Z axis: up is positive, layers (0: down, 1: up)
        # Layer below vehicle and layer above vehicle
        grid_z = int((z + self.cell_size_z) / self.cell_size_z)

        return grid_x, grid_y, grid_z

    def _in_bounds(self, grid_x, grid_y, grid_z):
        """
        Check if grid indices are within bounds.

        Args:
            grid_x, grid_y, grid_z (int): Grid indices

        Returns:
            bool: True if within bounds, False otherwise
        """
        return (
            0 <= grid_x < self.grid_dim_xy
            and 0 <= grid_y < self.grid_dim_xy
            and 0 <= grid_z < self.grid_dim_z
        )

    def _generate_class_colors(self):
        """
        Generate distinct colors for each object class.

        Returns:
            list: List of ColorRGBA objects for each class
        """
        colors = []
        for i in range(max(self.num_classes, 8)):  # At least 8 colors
            hue = (i * 360.0 / max(self.num_classes, 8)) / 360.0
            # Convert HSV to RGB (simplified)
            c = ColorRGBA()
            if hue < 1.0 / 6.0:
                c.r, c.g, c.b = 1.0, hue * 6.0, 0.0
            elif hue < 2.0 / 6.0:
                c.r, c.g, c.b = (2.0 / 6.0 - hue) * 6.0, 1.0, 0.0
            elif hue < 3.0 / 6.0:
                c.r, c.g, c.b = 0.0, 1.0, (hue - 2.0 / 6.0) * 6.0
            elif hue < 4.0 / 6.0:
                c.r, c.g, c.b = 0.0, (4.0 / 6.0 - hue) * 6.0, 1.0
            elif hue < 5.0 / 6.0:
                c.r, c.g, c.b = (hue - 4.0 / 6.0) * 6.0, 0.0, 1.0
            else:
                c.r, c.g, c.b = 1.0, 0.0, (6.0 / 6.0 - hue) * 6.0
            c.a = 0.7
            colors.append(c)
        return colors

    def _visualize_grid(self, grid, base_frame):
        """
        Visualize the 3D voxel grid as RViz markers.

        Args:
            grid (np.ndarray): The voxel grid
            base_frame (str): Reference frame for markers
        """
        marker_array = MarkerArray()

        # Grid boundary marker
        boundary_marker = Marker()
        boundary_marker.header.frame_id = base_frame
        boundary_marker.header.stamp = rospy.Time.now()
        boundary_marker.ns = "grid_boundary"
        boundary_marker.id = 0
        boundary_marker.type = Marker.LINE_LIST
        boundary_marker.action = Marker.ADD
        boundary_marker.scale.x = 0.02  # Line width
        boundary_marker.color = ColorRGBA(0.5, 0.5, 0.5, 0.5)

        # Draw grid boundary lines
        for layer in range(self.grid_dim_z):
            z_offset = -self.cell_size_z + layer * self.cell_size_z
            # Horizontal lines
            for i in range(self.grid_dim_xy + 1):
                offset = -self.grid_range_xy + i * self.cell_size_xy
                # X-aligned lines
                p1, p2 = Point(), Point()
                p1.x, p1.y, p1.z = self.grid_range_xy, offset, z_offset
                p2.x, p2.y, p2.z = -self.grid_range_xy, offset, z_offset
                boundary_marker.points.extend([p1, p2])
                # Y-aligned lines
                p1, p2 = Point(), Point()
                p1.x, p1.y, p1.z = offset, self.grid_range_xy, z_offset
                p2.x, p2.y, p2.z = offset, -self.grid_range_xy, z_offset
                boundary_marker.points.extend([p1, p2])

        marker_array.markers.append(boundary_marker)

        # Occupied cell markers
        marker_id = 1
        for y in range(self.grid_dim_xy):
            for x in range(self.grid_dim_xy):
                for z in range(self.grid_dim_z):
                    # Check if any object class is in this cell
                    cell_values = grid[y, x, z, :]
                    max_val = np.max(cell_values)

                    if max_val > 0.01:  # Threshold for visualization
                        max_class = np.argmax(cell_values)

                        marker = Marker()
                        marker.header.frame_id = base_frame
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "grid_cells"
                        marker.id = marker_id
                        marker.type = Marker.CUBE
                        marker.action = Marker.ADD

                        # Position in world coordinates
                        world_x = self.grid_range_xy - (y + 0.5) * self.cell_size_xy
                        world_y = -self.grid_range_xy + (x + 0.5) * self.cell_size_xy
                        world_z = -self.cell_size_z + (z + 0.5) * self.cell_size_z

                        marker.pose.position.x = world_x
                        marker.pose.position.y = world_y
                        marker.pose.position.z = world_z
                        marker.pose.orientation.w = 1.0

                        marker.scale.x = self.cell_size_xy * 0.9
                        marker.scale.y = self.cell_size_xy * 0.9
                        marker.scale.z = self.cell_size_z * 0.9

                        # Color based on object class
                        color = self.class_colors[max_class]
                        marker.color = ColorRGBA(
                            color.r, color.g, color.b, max_val * 0.6
                        )

                        marker_array.markers.append(marker)
                        marker_id += 1

        self.grid_pub.publish(marker_array)

    def _visualize_objects(self, object_positions, base_frame):
        """
        Visualize detected objects as markers.

        Args:
            object_positions (list): List of object position dictionaries
            base_frame (str): Reference frame for markers
        """
        marker_array = MarkerArray()

        for i, obj in enumerate(object_positions):
            # Object sphere marker
            marker = Marker()
            marker.header.frame_id = base_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "objects"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = obj["x"]
            marker.pose.position.y = obj["y"]
            marker.pose.position.z = obj["z"]
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.3

            color = self.class_colors[obj["class_idx"]]
            marker.color = ColorRGBA(color.r, color.g, color.b, 1.0)

            marker_array.markers.append(marker)

            # Text label
            text_marker = Marker()
            text_marker.header.frame_id = base_frame
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "object_labels"
            text_marker.id = i + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD

            text_marker.pose.position.x = obj["x"]
            text_marker.pose.position.y = obj["y"]
            text_marker.pose.position.z = obj["z"] + 0.3

            text_marker.scale.z = 0.15
            text_marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text_marker.text = f"{obj['frame'].split('/')[-1]}\nGrid:[{obj['grid_x']},{obj['grid_y']},{obj['grid_z']}]\nW:{obj['weight']:.2f}"

            marker_array.markers.append(text_marker)

        self.object_pub.publish(marker_array)

    def get_grid_info(self):
        """
        Get information about the grid configuration.

        Returns:
            dict: Grid configuration parameters
        """
        return {
            "grid_dimensions": (self.grid_dim_xy, self.grid_dim_xy, self.grid_dim_z),
            "cell_size_xy": self.cell_size_xy,
            "cell_size_z": self.cell_size_z,
            "coverage_xy": self.grid_range_xy * 2,
            "coverage_z": self.grid_range_z,
            "num_object_classes": self.num_classes,
            "object_frames": self.object_frames,
            "visualization": self.visualize,
        }

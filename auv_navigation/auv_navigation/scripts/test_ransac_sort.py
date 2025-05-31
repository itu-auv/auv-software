#!/usr/bin/env python
import numpy as np
import random
import matplotlib.pyplot as plt


# --- Mock ROS-like structures ---
class MockPoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Point(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f})"


class MockDetectedPipe:
    def __init__(self, x, y, z, color, pipe_id=0):
        self.position = MockPoint(x, y, z)
        self.color = color
        self.id = pipe_id  # Just for easier tracking in prints

    def __repr__(self):
        return f"Pipe(id={self.id}, color='{self.color}', pos={self.position})"


# Not strictly needed for this test as cluster_pipes_with_ransac takes a list of pipes
# class MockDetectedPipes:
#     def __init__(self):
#         self.detected_pipes = []
#         self.header = None # Mock header if needed


# --- Gate Class (from original code, slightly simplified for standalone) ---
class Gate:
    def __init__(self, white_left, red, white_right, direction):
        self.white_left = white_left
        self.red = red
        self.white_right = white_right
        self.direction = direction


# --- SlalomProcessorNode (modified for testing) ---
class SlalomProcessorNodeTest:
    def __init__(
        self,
        ransac_iterations=100,  # Reduced for faster testing
        line_distance_threshold=0.1,
        min_pipe_cluster_size=2,
        gate_angle_tolerance_degrees=15.0,  # Added
        # Removed other ROS-specific params for this test
    ):
        # --- Parameters (directly set for testing) ---
        self.ransac_iterations = ransac_iterations
        self.line_distance_threshold = line_distance_threshold
        self.min_pipe_cluster_size = min_pipe_cluster_size
        self.gate_angle_tolerance_degrees = gate_angle_tolerance_degrees  # Added
        self.gate_angle_cos_threshold = np.cos(
            np.deg2rad(self.gate_angle_tolerance_degrees)
        )  # Added

        # Removed TF, Subscribers, Publishers for this test

        # For easier debugging of pipe selection
        self.pipe_id_counter = 0

    # We are not testing filter_pipes_within_distance or get_pipe_distance_from_base
    # as they require TF. We will directly pass pipe lists to cluster_pipes_with_ransac.

    # --- System 1 functions ---
    def cluster_pipes_with_ransac(
        self, pipes, robot_y_axis_odom_2d=None
    ):  # Added robot_y_axis_odom_2d
        unassigned_pipes = list(pipes)
        gate_clusters = []

        perform_directional_check = robot_y_axis_odom_2d is not None

        while len(unassigned_pipes) >= self.min_pipe_cluster_size:
            best_inliers = []
            best_line_model = None

            # Ensure enough pipes for sampling
            if len(unassigned_pipes) < 2:
                break

            for _ in range(self.ransac_iterations):
                # Pick two unique pipes to define a line
                # random.sample requires population size >= k
                if len(unassigned_pipes) < 2:
                    break
                pipe_a, pipe_b = random.sample(unassigned_pipes, 2)
                pos_a = np.array([pipe_a.position.x, pipe_a.position.y])
                pos_b = np.array([pipe_b.position.x, pipe_b.position.y])
                line_vec = pos_b - pos_a
                length = np.linalg.norm(line_vec)
                if length < 1e-6:  # Avoid division by zero for coincident points
                    continue
                unit_direction = line_vec / length

                if perform_directional_check:
                    # Check if the gate line is parallel to the robot's Y-axis
                    # unit_direction is the direction of the potential gate line
                    # robot_y_axis_odom_2d is the expected orientation of gates
                    dot_product = np.abs(np.dot(unit_direction, robot_y_axis_odom_2d))
                    # abs(dot_product) should be close to 1 if parallel (cos(0) or cos(180))
                    # So, abs(dot_product) should be >= cos(tolerance_angle)
                    if dot_product < self.gate_angle_cos_threshold:
                        continue  # Line is not aligned with robot's Y-axis, skip this sample

                inliers = []
                for candidate_pipe in unassigned_pipes:
                    candidate_pos = np.array(
                        [candidate_pipe.position.x, candidate_pipe.position.y]
                    )
                    # Vector from line origin (pos_a) to candidate point
                    vec_ap = candidate_pos - pos_a
                    # Distance to line: || vec_ap - (vec_ap . unit_direction) * unit_direction ||
                    # Or, simpler: | (x2-x1)(y1-y0) - (x1-x0)(y2-y1) | / sqrt((x2-x1)^2 + (y2-y1)^2)
                    # For a line defined by (x1,y1) and direction (dx, dy) = unit_direction
                    # and point (x0,y0) = candidate_pos
                    # distance = |dx * (pos_a[1] - candidate_pos[1]) - dy * (pos_a[0] - candidate_pos[0])|
                    # where dx = unit_direction[0], dy = unit_direction[1]
                    # This is equivalent to cross product magnitude in 2D
                    distance_to_line = np.abs(
                        unit_direction[0] * (pos_a[1] - candidate_pos[1])
                        - unit_direction[1] * (pos_a[0] - candidate_pos[0])
                    )

                    if distance_to_line < self.line_distance_threshold:
                        inliers.append(candidate_pipe)

                if len(inliers) > len(best_inliers):
                    best_inliers = inliers
                    best_line_model = (
                        pos_a,
                        unit_direction,
                    )  # Store origin and direction

            if len(best_inliers) < self.min_pipe_cluster_size:
                break  # Not enough inliers to form a new cluster

            gate_clusters.append({"pipes": best_inliers, "line_model": best_line_model})
            unassigned_pipes = [p for p in unassigned_pipes if p not in best_inliers]

        return gate_clusters

    # --- System 2 functions ---
    def sort_pipes_along_line(self, cluster):
        if not cluster or "line_model" not in cluster or cluster["line_model"] is None:
            print("Warning: Invalid cluster provided to sort_pipes_along_line.")
            return {"pipes": cluster.get("pipes", []), "line_model": None}

        origin_point, unit_direction = cluster["line_model"]
        projections = []
        for pipe in cluster["pipes"]:
            pipe_position_xy = np.array([pipe.position.x, pipe.position.y])
            # Project pipe_position_xy onto the line defined by origin_point and unit_direction
            # The scalar projection is (pipe_position_xy - origin_point) . unit_direction
            projection_scalar = np.dot(
                (pipe_position_xy - origin_point), unit_direction
            )
            projections.append((projection_scalar, pipe))

        projections.sort(key=lambda item: item[0])
        sorted_pipes = [pipe for (_, pipe) in projections]
        return {"pipes": sorted_pipes, "line_model": cluster["line_model"]}

    # --- Helper to create pipes for testing ---
    def _create_pipe(self, x, y, color):
        pipe = MockDetectedPipe(x, y, 0.0, color, self.pipe_id_counter)
        self.pipe_id_counter += 1
        return pipe


# --- Test Data Generation ---
def generate_realistic_slalom_pipes(
    processor_node_instance,
    num_gates=3,
    inter_pipe_dist=1.5,  # Distance from red to white pipe in a gate
    gate_progression_x=4.0,  # How far along X for the next gate
    slalom_offset_y=2.0,  # Lateral Y offset for slalom
    initial_gate_center_x=0.0,
    initial_gate_center_y=0.0,
    gate_angle_rad_range=(-0.15, 0.15),  # Radians, tilt from horizontal
    position_noise_std=0.08,  # Std dev for Gaussian noise on x,y
    num_outliers=3,
):
    pipes = []

    for i in range(num_gates):
        # Determine gate center
        gc_x = initial_gate_center_x + i * gate_progression_x
        gc_y = initial_gate_center_y
        if i % 2 == 1:  # Apply slalom offset for odd-numbered gates (0-indexed)
            gc_y += slalom_offset_y

        gate_center = np.array([gc_x, gc_y])

        # Determine gate orientation
        angle = random.uniform(gate_angle_rad_range[0], gate_angle_rad_range[1])
        gate_direction_vec = np.array(
            [np.cos(angle), np.sin(angle)]
        )  # Unit vector along the gate

        # Pipe positions (ideal) relative to gate center along its direction
        # Red pipe is at the gate center
        red_pos_ideal = gate_center
        wl_pos_ideal = gate_center - gate_direction_vec * inter_pipe_dist
        wr_pos_ideal = gate_center + gate_direction_vec * inter_pipe_dist

        # Add noise and create pipes
        noise_wl = np.random.normal(0, position_noise_std, size=2)
        noise_r = np.random.normal(0, position_noise_std, size=2)
        noise_wr = np.random.normal(0, position_noise_std, size=2)

        pipes.append(
            processor_node_instance._create_pipe(
                wl_pos_ideal[0] + noise_wl[0], wl_pos_ideal[1] + noise_wl[1], "white"
            )
        )
        pipes.append(
            processor_node_instance._create_pipe(
                red_pos_ideal[0] + noise_r[0], red_pos_ideal[1] + noise_r[1], "red"
            )
        )
        pipes.append(
            processor_node_instance._create_pipe(
                wr_pos_ideal[0] + noise_wr[0], wr_pos_ideal[1] + noise_wr[1], "white"
            )
        )

    # Add some outliers
    # Determine bounds for outliers based on generated gates
    if pipes:
        all_x = [p.position.x for p in pipes]
        all_y = [p.position.y for p in pipes]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        x_range = max_x - min_x
        y_range = max_y - min_y
    else:  # Default if no gates were generated
        min_x, max_x, min_y, max_y = -5, 5, -5, 5
        x_range, y_range = 10, 10

    for _ in range(num_outliers):
        ox = random.uniform(min_x - x_range * 0.2, max_x + x_range * 0.2)
        oy = random.uniform(min_y - y_range * 0.2, max_y + y_range * 0.2)
        pipes.append(
            processor_node_instance._create_pipe(
                ox, oy, random.choice(["blue", "green", "yellow"])
            )
        )

    random.shuffle(pipes)
    return pipes


# NEW FUNCTION to generate pipes from user-specified coordinates
def generate_user_input_pipes(
    processor_node_instance, white_coords_list, red_coords_list
):
    """Generates a list of MockDetectedPipe objects from user-defined coordinates."""
    pipes = []
    # The pipe_id_counter should be managed by the caller if it needs to be reset globally
    # for the processor_node_instance. This function just uses the instance.
    for x, y in white_coords_list:
        pipes.append(processor_node_instance._create_pipe(x, y, "white"))
    for x, y in red_coords_list:
        pipes.append(processor_node_instance._create_pipe(x, y, "red"))
    # Shuffle the pipes, similar to the other generation function.
    # 'random' is imported at the top of the file.
    random.shuffle(pipes)
    return pipes


def generate_pipe_positions(num_gates=3, gate_spacing=1):
    """
    Generate pseudo-random positions for white and red PVC pipes in a slalom course.
    The gates generated are vertical (constant x for pipes in a gate).
    The robot's Y-axis is assumed to be parallel to these gate lines.

    Args:
        num_gates (int): Number of gates (default=3).
        gate_spacing (float): Approximate spacing between consecutive gates along the x-axis.

    Returns:
        white_pipe_coords (list of tuple): [(x1, y1), ..., (xN, yN)]
        red_pipe_coords   (list of tuple): [(x1, y1), ..., (xM, yM)]
        robot_y_axis_odom_2d (np.array): Assumed robot Y-axis direction in odom [0.0, 1.0]
    """
    white_pipe_coords = []
    red_pipe_coords = []

    for i in range(num_gates):
        # Center of this gate, with ±0.2 m noise along x
        x_center = i * gate_spacing + random.uniform(-0.2, 0.2)
        # Lateral center line of the red pipe, with ±0.2 m noise along y
        y_center = random.uniform(-0.2, 0.2)

        # Record red pipe
        red_pipe_coords.append((x_center, y_center))

        # White pipes are ±1.5 m from the red, plus a bit of noise
        y_left = y_center - 1.5 + random.uniform(-0.5, 0.5)
        y_right = y_center + 1.5 + random.uniform(-0.5, 0.5)

        white_pipe_coords.append((x_center, y_left))
        white_pipe_coords.append((x_center, y_right))

    # For gates aligned vertically (constant x), the robot's Y-axis, if looking straight,
    # would be parallel to the global Y-axis.
    robot_y_axis_odom_2d = np.array([0.0, 1.0])
    return white_pipe_coords, red_pipe_coords, robot_y_axis_odom_2d


# --- Visualization ---
def visualize_clusters(all_pipes, clusters, title="RANSAC Pipe Clustering"):
    plt.figure(figsize=(12, 9))  # Adjusted size for potentially more spread out data

    # Plot all original pipes
    # Pre-define colors for consistent legend
    pipe_color_map = {
        "red": "r",
        "white": "gray",
        "blue": "b",
        "green": "g",
        "yellow": "y",
    }
    plotted_legend_colors = set()

    for pipe in all_pipes:
        plot_color = pipe_color_map.get(pipe.color, "k")
        label = None
        if pipe.color not in plotted_legend_colors:
            label = f"Pipe ({pipe.color})"
            plotted_legend_colors.add(pipe.color)

        plt.scatter(
            pipe.position.x,
            pipe.position.y,
            c=plot_color,
            marker="o",
            s=100,
            label=label,
            alpha=0.4,
            edgecolors="k",
        )
        plt.text(
            pipe.position.x + 0.1, pipe.position.y + 0.1, f"{pipe.id}({pipe.color[0]})"
        )

    # Plot clustered pipes and their RANSAC lines
    for i, cluster_info in enumerate(clusters):
        cluster_pipes = cluster_info["pipes"]
        line_model = cluster_info["line_model"]

        # Use a distinct color for each cluster's line and inliers
        cluster_plot_color = plt.cm.rainbow(
            i / max(1, len(clusters))
        )  # Avoid division by zero if no clusters

        # Plot inlier pipes for this cluster
        for idx, pipe_in_cluster in enumerate(cluster_pipes):
            plt.scatter(
                pipe_in_cluster.position.x,
                pipe_in_cluster.position.y,
                c=[cluster_plot_color],
                marker="x",
                s=180,
                linewidths=1.5,
                label=f"Cluster {i} Inlier" if idx == 0 else None,
            )

        # Plot RANSAC line
        if line_model:
            origin, direction = line_model
            # Extend line for plotting based on typical gate span
            line_extension = 5.0  # How far to draw line beyond typical points

            # Find min/max projection of inliers to define segment, then extend
            if cluster_pipes:
                projs = [
                    np.dot(np.array([p.position.x, p.position.y]) - origin, direction)
                    for p in cluster_pipes
                ]
                min_proj, max_proj = min(projs), max(projs)
                p1 = origin + direction * (min_proj - line_extension * 0.5)
                p2 = origin + direction * (max_proj + line_extension * 0.5)
            else:  # Fallback if somehow no pipes in a cluster with a model
                p1 = origin - direction * line_extension
                p2 = origin + direction * line_extension

            plt.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                linestyle="--",
                color=cluster_plot_color,
                linewidth=2,
                label=f"Cluster {i} Line" if line_model else None,
            )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
    plt.tight_layout(rect=[0, 0, 0.83, 1])  # Adjust for legend
    plt.axis("equal")  # Important for correct perception of distances and angles
    plt.show()


# --- Main test execution ---
if __name__ == "__main__":
    processor = SlalomProcessorNodeTest(
        ransac_iterations=300,  # Might need more for noisy data or more outliers
        line_distance_threshold=0.25,  # Adjusted based on noise_std and potential gate tilt
        min_pipe_cluster_size=2,  # Finds pairs, which can then be validated to form gates
        gate_angle_tolerance_degrees=20.0,  # Added tolerance
    )

    # Reset pipe ID counter on the processor instance before generating pipes
    processor.pipe_id_counter = 0

    # Define user inputs for the new function
    # Get robot_y_axis from the pipe generation function
    white_pipe_coords, red_pipe_coords, robot_y_axis = generate_pipe_positions(
        num_gates=3,
        gate_spacing=1,  # Changed gate_spacing to 2 for better visualization
    )
    # Generate test pipes using the new function for user-defined inputs
    all_test_pipes = generate_user_input_pipes(
        processor, white_pipe_coords, red_pipe_coords
    )

    print("--- Generated Test Pipes (User Defined) ---")  # Updated print statement
    for p in all_test_pipes:
        print(p)
    print(f"Total pipes generated: {len(all_test_pipes)}")
    print(f"Assumed Robot Y-axis for directional check: {robot_y_axis}")
    print("-" * 30)

    print("\\n--- Testing cluster_pipes_with_ransac ---")
    # Pass the robot_y_axis to RANSAC
    raw_clusters = processor.cluster_pipes_with_ransac(
        all_test_pipes, robot_y_axis_odom_2d=robot_y_axis
    )

    print(f"Found {len(raw_clusters)} clusters.")
    for i, cluster in enumerate(raw_clusters):
        print(f"Cluster {i}:")
        # Ensure line_model exists before trying to access its elements
        if cluster["line_model"]:
            origin_str = f"({cluster['line_model'][0][0]:.2f}, {cluster['line_model'][0][1]:.2f})"
            direction_str = f"({cluster['line_model'][1][0]:.2f}, {cluster['line_model'][1][1]:.2f})"
            print(f"  Line Model (origin: {origin_str}, direction: {direction_str})")
        else:
            print("  Line Model: None")
        print(f"  Pipes in cluster ({len(cluster['pipes'])}):")
        for pipe in cluster["pipes"]:
            print(f"    {pipe}")
    print("-" * 30)

    visualize_clusters(
        all_test_pipes, raw_clusters, title="RANSAC Pipe Clustering (Realistic Slalom)"
    )

    print("\n--- Testing sort_pipes_along_line for each cluster ---")
    sorted_clusters_data = []
    for i, cluster in enumerate(raw_clusters):
        print(f"Sorting Cluster {i} (IDs): {[p.id for p in cluster['pipes']]}")

        sorted_cluster_info = processor.sort_pipes_along_line(cluster)
        sorted_clusters_data.append(sorted_cluster_info)

        print(
            f"  Sorted Pipe IDs in Cluster {i}: {[p.id for p in sorted_cluster_info['pipes']]}"
        )
        # For detailed check:
        # for pipe in sorted_cluster_info['pipes']:
        #     print(f"    {pipe}")
        print("-" * 20)
    print("-" * 30)

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
        # Removed other ROS-specific params for this test
    ):
        # --- Parameters (directly set for testing) ---
        self.ransac_iterations = ransac_iterations
        self.line_distance_threshold = line_distance_threshold
        self.min_pipe_cluster_size = min_pipe_cluster_size

        # Removed TF, Subscribers, Publishers for this test

        # For easier debugging of pipe selection
        self.pipe_id_counter = 0

    # We are not testing filter_pipes_within_distance or get_pipe_distance_from_base
    # as they require TF. We will directly pass pipe lists to cluster_pipes_with_ransac.

    # --- System 1 functions ---
    def cluster_pipes_with_ransac(self, pipes):
        unassigned_pipes = list(pipes)
        gate_clusters = []

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
def generate_test_pipes(processor_node_instance):
    pipes = []
    # Gate 1 (approx y=1)
    pipes.append(processor_node_instance._create_pipe(0, 1.0, "white"))
    pipes.append(processor_node_instance._create_pipe(1.5, 1.05, "red"))
    pipes.append(processor_node_instance._create_pipe(3.0, 0.95, "white"))
    pipes.append(
        processor_node_instance._create_pipe(1.4, 2.5, "blue")
    )  # Nearby outlier for Gate 1

    # Gate 2 (approx x=4)
    pipes.append(processor_node_instance._create_pipe(4.0, 0, "white"))
    pipes.append(processor_node_instance._create_pipe(4.05, 1.5, "red"))
    pipes.append(processor_node_instance._create_pipe(3.95, 3.0, "white"))

    # Outliers
    pipes.append(processor_node_instance._create_pipe(-2, -2, "green"))
    pipes.append(processor_node_instance._create_pipe(5, 5, "yellow"))

    # A small cluster of 2 that might be picked up
    pipes.append(processor_node_instance._create_pipe(0, 4.0, "white"))
    pipes.append(processor_node_instance._create_pipe(0.5, 4.05, "red"))

    random.shuffle(pipes)  # Shuffle to make it more realistic
    return pipes


# --- Visualization ---
def visualize_clusters(all_pipes, clusters, title="RANSAC Pipe Clustering"):
    plt.figure(figsize=(10, 8))

    # Plot all original pipes
    for pipe in all_pipes:
        color_map = {
            "red": "r",
            "white": "gray",
            "blue": "b",
            "green": "g",
            "yellow": "y",
        }
        plt.scatter(
            pipe.position.x,
            pipe.position.y,
            c=color_map.get(pipe.color, "k"),
            marker="o",
            s=100,
            label=f"All Pipes ({pipe.color})" if pipe.id == 0 else None,
            alpha=0.3,
        )
        plt.text(
            pipe.position.x + 0.1, pipe.position.y + 0.1, f"{pipe.id}({pipe.color[0]})"
        )

    # Plot clustered pipes and their RANSAC lines
    for i, cluster_info in enumerate(clusters):
        cluster_pipes = cluster_info["pipes"]
        line_model = cluster_info["line_model"]

        cluster_color_plot = plt.cm.viridis(
            i / len(clusters) if len(clusters) > 0 else 0
        )

        # Plot inlier pipes for this cluster
        for pipe in cluster_pipes:
            plt.scatter(
                pipe.position.x,
                pipe.position.y,
                c=[cluster_color_plot],  # Use a consistent color for the cluster
                marker="x",
                s=150,
                label=f"Cluster {i} Inlier" if pipe == cluster_pipes[0] else None,
            )

        # Plot RANSAC line
        if line_model:
            origin, direction = line_model
            # Extend line for plotting
            p1 = origin - direction * 5  # Extend backward
            p2 = origin + direction * 10  # Extend forward
            plt.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                linestyle="--",
                color=cluster_color_plot,
                label=f"Cluster {i} Line" if line_model else None,
            )

    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)

    # Collect unique labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicate labels
    if by_label:  # only show legend if there are labels
        plt.legend(
            by_label.values(),
            by_label.keys(),
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    plt.axis("equal")
    plt.show()


# --- Main test execution ---
if __name__ == "__main__":
    # 1. Initialize the test version of SlalomProcessorNode
    # Increased iterations for potentially better RANSAC, decreased for speed in testing
    # Increased tolerance for slightly more spread out points
    processor = SlalomProcessorNodeTest(
        ransac_iterations=200,
        line_distance_threshold=0.2,
        min_pipe_cluster_size=2,  # Should be at least 2 for RANSAC
    )

    # 2. Generate test pipes
    all_test_pipes = generate_test_pipes(processor)
    print("--- Generated Test Pipes ---")
    for p in all_test_pipes:
        print(p)
    print("-" * 30)

    # 3. Test cluster_pipes_with_ransac
    print("\n--- Testing cluster_pipes_with_ransac ---")
    raw_clusters = processor.cluster_pipes_with_ransac(all_test_pipes)

    print(f"Found {len(raw_clusters)} clusters.")
    for i, cluster in enumerate(raw_clusters):
        print(f"Cluster {i}:")
        print(f"  Line Model (origin, direction): {cluster['line_model']}")
        print(f"  Pipes in cluster ({len(cluster['pipes'])}):")
        for pipe in cluster["pipes"]:
            print(f"    {pipe}")
    print("-" * 30)

    # Visualize the raw clusters
    visualize_clusters(
        all_test_pipes, raw_clusters, title="RANSAC Pipe Clustering Results"
    )

    # 4. Test sort_pipes_along_line for each found cluster
    print("\n--- Testing sort_pipes_along_line for each cluster ---")
    sorted_clusters_data = []
    for i, cluster in enumerate(raw_clusters):
        print(f"Sorting Cluster {i}:")
        print(
            f"  Pipes BEFORE sorting ({len(cluster['pipes'])}): {[p.id for p in cluster['pipes']]}"
        )

        sorted_cluster_info = processor.sort_pipes_along_line(cluster)
        sorted_clusters_data.append(
            sorted_cluster_info
        )  # Store for potential further use/viz

        print(f"  Pipes AFTER sorting ({len(sorted_cluster_info['pipes'])}):")
        for pipe in sorted_cluster_info["pipes"]:
            print(f"    {pipe}")
        print("-" * 20)
    print("-" * 30)

    # Note: Visualization for sort_pipes_along_line is implicitly covered by
    # the RANSAC visualization if you mentally trace the line.
    # Explicitly showing sort order might involve numbering pipes in the plot
    # or printing their order clearly, which we've done above.

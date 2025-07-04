#!/usr/bin/env python
import rospy
import numpy as np
import random
from auv_msgs.msg import Pipes, Pipe
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray


class SlalomPipePublisherNode:
    def __init__(self):
        rospy.init_node("slalom_pipe_publisher")

        # --- Parameters for Pipe Generation (from generate_realistic_slalom_pipes) ---
        self.num_gates = rospy.get_param("~num_gates", 3)
        self.inter_pipe_dist = rospy.get_param("~inter_pipe_dist", 1.5)
        self.gate_progression_x = rospy.get_param("~gate_progression_x", 4.0)
        self.slalom_offset_y = rospy.get_param("~slalom_offset_y", 1.0)
        self.initial_gate_center_x = rospy.get_param("~initial_gate_center_x", 0.0)
        self.initial_gate_center_y = rospy.get_param("~initial_gate_center_y", 0.0)
        gate_angle_min_rad = rospy.get_param("~gate_angle_min_rad", -0.15)
        gate_angle_max_rad = rospy.get_param("~gate_angle_max_rad", 0.15)
        self.gate_angle_rad_range = (gate_angle_min_rad, gate_angle_max_rad)
        self.position_noise_std = rospy.get_param("~position_noise_std", 0.08)
        self.num_outliers = rospy.get_param("~num_outliers", 3)
        self.pipe_z_position = rospy.get_param(
            "~pipe_z_position", 0.0
        )  # Added for Z control

        # --- Publisher ---
        self.pipes_publisher = rospy.Publisher("/slalom_pipes", Pipes, queue_size=10)
        self.marker_publisher = rospy.Publisher(
            "/slalom_pipes_markers", MarkerArray, queue_size=10
        )
        self.pipe_frame_id = rospy.get_param("~pipe_frame_id", "odom")

        # --- Generate pipes once at startup ---
        self.static_pipes = []
        # Red pipes
        red_pipe_coords = [(2.0, 0.0), (3.5, -0.3), (5.0, 0.0)]
        for x, y in red_pipe_coords:
            self.static_pipes.append(
                self._create_ros_pipe(x, y, self.pipe_z_position, "red")
            )

        # White pipes
        white_pipe_coords = [
            (2.0, -1.5),
            (2.0, 1.5),
            (3.5, -1.8),
            (3.5, 1.2),
            (5.0, -1.5),
            (5.0, 1.5),
        ]
        for x, y in white_pipe_coords:
            self.static_pipes.append(
                self._create_ros_pipe(x, y, self.pipe_z_position, "white")
            )

        rospy.loginfo(f"Initialized with {len(self.static_pipes)} static pipes.")
        rospy.loginfo(f"Generated {len(self.static_pipes)} pipes at startup.")

        # --- Timer for continuous publishing ---
        self.publish_rate_hz = rospy.get_param(
            "~publish_rate_hz", 1.0
        )  # Store for marker lifetime
        if self.publish_rate_hz <= 0:
            rospy.logwarn(
                "Publish rate set to 0 or negative. Publisher will not run periodically."
            )
        else:
            rospy.Timer(
                rospy.Duration(1.0 / self.publish_rate_hz), self.publish_pipes_callback
            )

        rospy.loginfo(
            "Slalom Pipe Publisher node started. Publishing to /slalom_pipes and /slalom_pipes_markers"
        )

    def _create_ros_pipe(self, x: float, y: float, z: float, color_str: str) -> Pipe:
        """Helper to create an auv_msgs.msg.Pipe object."""
        pipe = Pipe()
        pipe.color = color_str
        pipe.position.x = float(x)
        pipe.position.y = float(y)
        pipe.position.z = float(z)
        return pipe

    def generate_pipes_data(self):
        """
        Adapted from generate_realistic_slalom_pipes.
        Generates a list of auv_msgs.msg.Pipe objects.
        """
        pipes = []

        for i in range(self.num_gates):
            gc_x = self.initial_gate_center_x + i * self.gate_progression_x
            gc_y = self.initial_gate_center_y
            if i % 2 == 1:  # Apply slalom offset for odd-numbered gates
                gc_y += self.slalom_offset_y
            gate_center = np.array([gc_x, gc_y])

            angle = random.uniform(
                self.gate_angle_rad_range[0], self.gate_angle_rad_range[1]
            )
            gate_direction_vec = np.array([np.cos(angle), np.sin(angle)])

            red_pos_ideal = gate_center
            wl_pos_ideal = gate_center - gate_direction_vec * self.inter_pipe_dist
            wr_pos_ideal = gate_center + gate_direction_vec * self.inter_pipe_dist

            noise_wl = np.random.normal(0, self.position_noise_std, size=2)
            noise_r = np.random.normal(0, self.position_noise_std, size=2)
            noise_wr = np.random.normal(0, self.position_noise_std, size=2)

            pipes.append(
                self._create_ros_pipe(
                    wl_pos_ideal[0] + noise_wl[0],
                    wl_pos_ideal[1] + noise_wl[1],
                    self.pipe_z_position,
                    "white",
                )
            )
            pipes.append(
                self._create_ros_pipe(
                    red_pos_ideal[0] + noise_r[0],
                    red_pos_ideal[1] + noise_r[1],
                    self.pipe_z_position,
                    "red",
                )
            )
            pipes.append(
                self._create_ros_pipe(
                    wr_pos_ideal[0] + noise_wr[0],
                    wr_pos_ideal[1] + noise_wr[1],
                    self.pipe_z_position,
                    "white",
                )
            )

        if pipes:
            all_x = [p.position.x for p in pipes]
            all_y = [p.position.y for p in pipes]
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            x_range = max_x - min_x if max_x > min_x else 1.0  # Avoid zero range
            y_range = max_y - min_y if max_y > min_y else 1.0  # Avoid zero range
        else:
            min_x, max_x, min_y, max_y = -5.0, 5.0, -5.0, 5.0
            x_range, y_range = 10.0, 10.0

        for _ in range(self.num_outliers):
            ox = random.uniform(min_x - x_range * 0.2, max_x + x_range * 0.2)
            oy = random.uniform(min_y - y_range * 0.2, max_y + y_range * 0.2)
            pipes.append(
                self._create_ros_pipe(
                    ox,
                    oy,
                    self.pipe_z_position,
                    random.choice(
                        ["blue", "yellow", "black"]
                    ),  # Avoid green if used for something else
                )
            )

        random.shuffle(pipes)
        return pipes

    def publish_pipes_callback(self, event=None):
        # Use the pre-generated static pipes
        list_of_pipe_objects = self.static_pipes

        if not list_of_pipe_objects:
            rospy.logwarn_throttle(
                10, "No static pipes generated or available to publish."
            )
            return

        msg = Pipes()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.pipe_frame_id
        msg.pipes = list_of_pipe_objects

        self.pipes_publisher.publish(msg)
        # rospy.loginfo(f"Published {len(list_of_pipe_objects)} pipes to /slalom_pipes")

        # --- Publish Markers ---
        marker_array = MarkerArray()
        marker_id_counter = 0
        for pipe_obj in list_of_pipe_objects:
            if pipe_obj.color not in ["red", "white"]:
                continue  # Skip markers for other colors

            marker = Marker()
            marker.header.frame_id = self.pipe_frame_id
            marker.header.stamp = msg.header.stamp  # Use same timestamp
            marker.ns = "slalom_pipes"
            marker.id = marker_id_counter
            marker_id_counter += 1
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = pipe_obj.position.x
            marker.pose.position.y = pipe_obj.position.y
            marker.pose.position.z = pipe_obj.position.z  # Center of cylinder at pipe_z
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.1  # Diameter
            marker.scale.y = 0.1  # Diameter
            marker.scale.z = 1.0  # Height of the cylinder

            if pipe_obj.color == "red":
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            elif pipe_obj.color == "white":
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
            # No other colors needed as per user feedback
            marker.color.a = 0.8  # Alpha

            if self.publish_rate_hz > 0:
                marker.lifetime = rospy.Duration(
                    1.5 / self.publish_rate_hz
                )  # Make them last a bit longer than publish interval
            else:
                marker.lifetime = rospy.Duration(
                    0
                )  # Last forever if not periodically published

            marker_array.markers.append(marker)

        if marker_array.markers:
            self.marker_publisher.publish(marker_array)
            # rospy.loginfo(f"Published {len(marker_array.markers)} markers to /slalom_pipes_markers")


if __name__ == "__main__":
    try:
        SlalomPipePublisherNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

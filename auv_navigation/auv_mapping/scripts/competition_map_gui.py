#!/usr/bin/env python3

import tkinter as tk
from tkinter import messagebox
import math

import rospy
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_euler
from auv_msgs.msg import ObjectPose
from auv_msgs.srv import SetPremap, SetPremapRequest


class CompetitionMapGUI:
    POOL_WIDTH = 50.0
    POOL_HEIGHT = 25.0

    OBJECTS = [
        ("reference", "green", "frame", 0),
        ("gate", "red", "line", 3.0),
        ("bin", "purple", "point", 0),
        ("torpedo", "blue", "line", 1.5),
        ("octagon", "orange", "point", 0),
    ]

    DEFAULT_Z = {
        "gate": -0.5,
        "bin": -2.0,
        "torpedo": -1.5,
        "octagon": -1.3,
    }

    SERVICE_LABEL_MAP = {
        "gate": ["gate_sawfish_link", "gate_shark_link"],
        "bin": ["bin_whole_link"],
        "torpedo": ["torpedo_map_link"],
        "octagon": ["octagon_link"],
    }

    REFERENCE_FRAME = "coin_flip_rescuer"
    REFERENCE_FRAME_OPTIONS = ("coin_flip_rescuer", "odom")

    def __init__(self, root):
        self.root = root
        self.root.title("Competition Pool Map")
        self.root.attributes("-zoomed", True)

        self.service_name = rospy.get_param("~set_premap_service", "map/p_set_premap")
        default_reference_frame = str(
            rospy.get_param("~reference_frame", self.REFERENCE_FRAME)
        )

        if default_reference_frame not in self.REFERENCE_FRAME_OPTIONS:
            rospy.logwarn(
                f"Invalid ~reference_frame='{default_reference_frame}', "
                f"falling back to '{self.REFERENCE_FRAME}'."
            )
            default_reference_frame = self.REFERENCE_FRAME

        self.reference_frame_var = tk.StringVar(value=default_reference_frame)
        self.selected_object = tk.StringVar(value="reference")
        self.placed_objects = {}
        self.canvas_padding = 40
        self.scale = 1.0
        self.pool_x_offset = 0
        self.pool_y_offset = 0
        self.service_connected = False

        self.setup_ui()
        self.root.after(100, self.draw_pool)
        self.root.after(500, self.check_ros_service)

    def check_ros_service(self):
        """Check if ROS service is available"""
        was_connected = self.service_connected
        try:
            rospy.wait_for_service(self.service_name, timeout=1.0)
            self.service_connected = True
            self.status_label.config(text="ROS: Connected", fg="green")
            if not was_connected:
                rospy.loginfo(f"Service connected: {self.service_name}")
        except Exception:
            self.service_connected = False
            self.status_label.config(text="ROS: Not connected", fg="red")
            if was_connected:
                rospy.logwarn(f"Service disconnected: {self.service_name}")

        self.root.after(5000, self.check_ros_service)

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.LabelFrame(main_frame, text="Objects", width=280)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="Select object:").pack(pady=(10, 5))

        for obj_name, obj_color, obj_type, _ in self.OBJECTS:
            rb = tk.Radiobutton(
                left_frame,
                text=obj_name,
                variable=self.selected_object,
                value=obj_name,
                fg=obj_color,
                anchor=tk.W,
            )
            rb.pack(fill=tk.X, padx=10, pady=2)

        tk.Label(left_frame, text="\nOrientation (frame/gate/torpedo):").pack(
            pady=(10, 5)
        )

        yaw_frame = tk.Frame(left_frame)
        yaw_frame.pack(fill=tk.X, padx=10)

        self.yaw_var = tk.DoubleVar(value=0)
        self.yaw_scale = tk.Scale(
            yaw_frame,
            from_=180,
            to=-180,
            orient=tk.HORIZONTAL,
            variable=self.yaw_var,
            label="Yaw (deg)",
            command=self.on_yaw_change,
        )
        self.yaw_scale.pack(fill=tk.X)

        tk.Button(
            left_frame, text="Reset Orientation (0°)", command=self.reset_orientation
        ).pack(fill=tk.X, padx=10, pady=5)

        tk.Frame(left_frame, height=2, bg="gray").pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            left_frame, text="Vehicle Communication:", font=("Arial", 9, "bold")
        ).pack(pady=(5, 5))

        tk.Label(left_frame, text="Reference frame:").pack(pady=(5, 2))
        tk.OptionMenu(
            left_frame, self.reference_frame_var, *self.REFERENCE_FRAME_OPTIONS
        ).pack(fill=tk.X, padx=10)

        tk.Button(
            left_frame,
            text="Send to Vehicle",
            command=self.send_to_vehicle,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(left_frame, text="ROS: Checking...", fg="orange")
        self.status_label.pack(pady=5)

        tk.Button(
            left_frame,
            text="Clear All",
            command=self.clear_all,
            bg="#f44336",
            fg="white",
        ).pack(fill=tk.X, padx=10, pady=10)

        self.coord_label = tk.Label(left_frame, text="Mouse: X: - Y: -")
        self.coord_label.pack(pady=5)

        tk.Frame(left_frame, height=2, bg="gray").pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            left_frame,
            text="Positions (relative to reference):",
            font=("Arial", 9, "bold"),
        ).pack(pady=(5, 5))

        self.positions_text = tk.Text(
            left_frame, height=25, width=32, font=("Courier", 9)
        )
        self.positions_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.positions_text.config(state=tk.DISABLED)

        right_frame = tk.LabelFrame(
            main_frame, text="Pool (50m x 25m) - Origin at bottom-left"
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, bg="lightblue")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Configure>", self.on_resize)

    def on_yaw_change(self, value):
        """Update the yaw of selected line/frame object if it's placed"""
        obj_name = self.selected_object.get()
        if obj_name in self.placed_objects:
            obj_data = self.placed_objects[obj_name]
            if obj_data[3] in ("line", "frame"):
                self.placed_objects[obj_name] = (
                    obj_data[0],
                    obj_data[1],
                    obj_data[2],
                    obj_data[3],
                    obj_data[4],
                    float(value),
                )
                self.redraw_objects()
                self.update_positions_text()

    def on_resize(self, event):
        self.draw_pool()
        self.redraw_objects()

    def draw_pool(self):
        self.canvas.delete("pool")

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w < 100 or canvas_h < 100:
            return

        available_w = canvas_w - 2 * self.canvas_padding
        available_h = canvas_h - 2 * self.canvas_padding

        scale_x = available_w / self.POOL_WIDTH
        scale_y = available_h / self.POOL_HEIGHT
        self.scale = min(scale_x, scale_y)

        pool_w = self.POOL_WIDTH * self.scale
        pool_h = self.POOL_HEIGHT * self.scale

        self.pool_x_offset = (canvas_w - pool_w) / 2
        self.pool_y_offset = (canvas_h - pool_h) / 2

        self.canvas.create_rectangle(
            self.pool_x_offset,
            self.pool_y_offset,
            self.pool_x_offset + pool_w,
            self.pool_y_offset + pool_h,
            outline="darkblue",
            width=3,
            fill="white",
            tags="pool",
        )

        for i in range(1, 10):
            y = i * 2.5
            py = self.pool_y_offset + pool_h - y * self.scale
            self.canvas.create_line(
                self.pool_x_offset,
                py,
                self.pool_x_offset + pool_w,
                py,
                fill="gray",
                dash=(2, 4),
                tags="pool",
            )
            self.canvas.create_text(
                self.pool_x_offset - 20,
                py,
                text=f"{y:.1f}",
                font=("Arial", 8),
                tags="pool",
            )

        y_extra = 1.5
        py_extra = self.pool_y_offset + pool_h - y_extra * self.scale
        self.canvas.create_line(
            self.pool_x_offset,
            py_extra,
            self.pool_x_offset + pool_w,
            py_extra,
            fill="red",
            dash=(4, 2),
            width=1,
            tags="pool",
        )
        self.canvas.create_text(
            self.pool_x_offset - 20,
            py_extra,
            text=f"{y_extra:.1f}",
            font=("Arial", 8),
            fill="red",
            tags="pool",
        )

        for i in range(1, 4):
            x = i * 12.5
            px = self.pool_x_offset + x * self.scale
            self.canvas.create_line(
                px,
                self.pool_y_offset,
                px,
                self.pool_y_offset + pool_h,
                fill="gray",
                dash=(2, 4),
                tags="pool",
            )
            self.canvas.create_text(
                px,
                self.pool_y_offset + pool_h + 15,
                text=f"{x:.1f}",
                font=("Arial", 8),
                tags="pool",
            )

        self.canvas.create_text(
            self.pool_x_offset,
            self.pool_y_offset + pool_h + 15,
            text="0",
            font=("Arial", 8),
            tags="pool",
        )
        self.canvas.create_text(
            self.pool_x_offset + pool_w,
            self.pool_y_offset + pool_h + 15,
            text="50m",
            font=("Arial", 8),
            tags="pool",
        )
        self.canvas.create_text(
            self.pool_x_offset - 20,
            self.pool_y_offset,
            text="25m",
            font=("Arial", 8),
            tags="pool",
        )
        self.canvas.create_text(
            self.pool_x_offset - 20,
            self.pool_y_offset + pool_h,
            text="0",
            font=("Arial", 8),
            tags="pool",
        )

    def meters_to_pixels(self, x_m, y_m):
        """Convert meters to pixels, origin at bottom-left"""
        pool_h = self.POOL_HEIGHT * self.scale
        px = self.pool_x_offset + x_m * self.scale
        py = self.pool_y_offset + pool_h - y_m * self.scale
        return px, py

    def pixels_to_meters(self, px, py):
        """Convert pixels to meters, origin at bottom-left"""
        pool_h = self.POOL_HEIGHT * self.scale
        x_m = (px - self.pool_x_offset) / self.scale
        y_m = (self.pool_y_offset + pool_h - py) / self.scale
        return x_m, y_m

    def on_mouse_move(self, event):
        x_m, y_m = self.pixels_to_meters(event.x, event.y)
        if 0 <= x_m <= self.POOL_WIDTH and 0 <= y_m <= self.POOL_HEIGHT:
            self.coord_label.config(text=f"Mouse: X: {x_m:.2f}m  Y: {y_m:.2f}m")
        else:
            self.coord_label.config(text="Mouse: X: -  Y: -")

    def on_click(self, event):
        x_m, y_m = self.pixels_to_meters(event.x, event.y)

        if not (0 <= x_m <= self.POOL_WIDTH and 0 <= y_m <= self.POOL_HEIGHT):
            return

        obj_name = self.selected_object.get()
        obj_color = "black"
        obj_type = "point"
        obj_length = 0

        for name, color, otype, length in self.OBJECTS:
            if name == obj_name:
                obj_color = color
                obj_type = otype
                obj_length = length
                break

        yaw = self.yaw_var.get() if obj_type in ("line", "frame") else 0
        self.placed_objects[obj_name] = (x_m, y_m, obj_color, obj_type, obj_length, yaw)
        self.redraw_objects()
        self.update_positions_text()

    def draw_single_object(self, name, x_m, y_m, color, obj_type, length, yaw):
        px, py = self.meters_to_pixels(x_m, y_m)

        if obj_type == "frame":
            arrow_len = 2.0 * self.scale
            yaw_rad = math.radians(yaw + 90)

            dx = arrow_len * math.cos(yaw_rad)
            dy = arrow_len * math.sin(yaw_rad)
            self.canvas.create_line(
                px,
                py,
                px + dx,
                py - dy,
                fill=color,
                width=3,
                arrow=tk.LAST,
                tags="object",
            )

            r = 6
            self.canvas.create_oval(
                px - r,
                py - r,
                px + r,
                py + r,
                fill="white",
                outline=color,
                width=2,
                tags="object",
            )

        elif obj_type == "line" and length > 0:
            half_len_px = (length / 2) * self.scale
            yaw_rad = math.radians(yaw + 90)

            perp_rad = yaw_rad - math.pi / 2
            dx = half_len_px * math.cos(perp_rad)
            dy = half_len_px * math.sin(perp_rad)

            x1, y1 = px - dx, py + dy
            x2, y2 = px + dx, py - dy

            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=4, tags="object")

            arrow_len = 1.5 * self.scale
            arrow_dx = arrow_len * math.cos(yaw_rad)
            arrow_dy = arrow_len * math.sin(yaw_rad)
            self.canvas.create_line(
                px,
                py,
                px + arrow_dx,
                py - arrow_dy,
                fill=color,
                width=2,
                arrow=tk.LAST,
                tags="object",
            )

            r = 5
            self.canvas.create_oval(
                px - r,
                py - r,
                px + r,
                py + r,
                fill=color,
                outline="black",
                width=1,
                tags="object",
            )
        else:
            r = 7
            self.canvas.create_oval(
                px - r,
                py - r,
                px + r,
                py + r,
                fill=color,
                outline="black",
                width=2,
                tags="object",
            )

        self.canvas.create_text(
            px, py + 20, text=name, font=("Arial", 9, "bold"), tags="object"
        )

    def redraw_objects(self):
        self.canvas.delete("object")
        for obj_name, obj_data in self.placed_objects.items():
            x_m, y_m, color, obj_type, length, yaw = obj_data
            self.draw_single_object(obj_name, x_m, y_m, color, obj_type, length, yaw)

    def update_positions_text(self):
        """Update the positions text showing all objects relative to frame"""
        self.positions_text.config(state=tk.NORMAL)
        self.positions_text.delete(1.0, tk.END)

        if "reference" not in self.placed_objects:
            self.positions_text.insert(
                tk.END, "Place reference first to see\nrelative positions."
            )
            self.positions_text.config(state=tk.DISABLED)
            return

        frame_data = self.placed_objects["reference"]
        frame_x, frame_y = frame_data[0], frame_data[1]
        frame_yaw = frame_data[5]
        frame_yaw_rad = math.radians(frame_yaw)

        self.positions_text.insert(tk.END, "REFERENCE (frame origin):\n")
        self.positions_text.insert(tk.END, "  x=0.00  y=0.00\n")
        self.positions_text.insert(tk.END, "  yaw=0.0°\n")
        self.positions_text.insert(
            tk.END,
            f"  anchor_abs: x={frame_x:.2f}  y={frame_y:.2f}  yaw={frame_yaw:.1f}°\n",
        )
        self.positions_text.insert(tk.END, "-" * 28 + "\n")
        self.positions_text.insert(tk.END, "Relative to reference:\n")

        for obj_name, obj_data in self.placed_objects.items():
            if obj_name == "reference":
                continue

            obj_x, obj_y = obj_data[0], obj_data[1]
            obj_type = obj_data[3]
            obj_yaw = obj_data[5]

            dx = obj_x - frame_x
            dy = obj_y - frame_y

            rel_x = dx * math.cos(-frame_yaw_rad) - dy * math.sin(-frame_yaw_rad)
            rel_y = dx * math.sin(-frame_yaw_rad) + dy * math.cos(-frame_yaw_rad)

            z_val = self.DEFAULT_Z.get(obj_name, 0.0)

            self.positions_text.insert(tk.END, f"\n{obj_name}:\n")
            self.positions_text.insert(tk.END, f"  x={rel_x:.2f}  y={rel_y:.2f}\n")
            self.positions_text.insert(tk.END, f"  z={z_val:.2f}\n")

            if obj_type in ("line", "frame"):
                rel_yaw = obj_yaw - frame_yaw
                while rel_yaw > 180:
                    rel_yaw -= 360
                while rel_yaw < -180:
                    rel_yaw += 360
                self.positions_text.insert(tk.END, f"  yaw={rel_yaw:.1f}°\n")

        self.positions_text.config(state=tk.DISABLED)

    def reset_orientation(self):
        """Reset yaw slider to 0 and update selected object if placed"""
        self.yaw_var.set(0)
        obj_name = self.selected_object.get()
        if obj_name in self.placed_objects:
            obj_data = self.placed_objects[obj_name]
            if obj_data[3] in ("line", "frame"):
                self.placed_objects[obj_name] = (
                    obj_data[0],
                    obj_data[1],
                    obj_data[2],
                    obj_data[3],
                    obj_data[4],
                    0.0,
                )
                self.redraw_objects()
                self.update_positions_text()

    def clear_all(self):
        self.placed_objects.clear()
        self.canvas.delete("object")
        self.positions_text.config(state=tk.NORMAL)
        self.positions_text.delete(1.0, tk.END)
        self.positions_text.config(state=tk.DISABLED)

    def get_relative_positions(self):
        """Calculate relative positions of all objects to reference frame"""
        if "reference" not in self.placed_objects:
            return None

        ref_data = self.placed_objects["reference"]
        ref_x, ref_y = ref_data[0], ref_data[1]
        ref_yaw = ref_data[5]
        ref_yaw_rad = math.radians(ref_yaw)

        result = {}

        for obj_name, obj_data in self.placed_objects.items():
            if obj_name == "reference":
                continue

            obj_x, obj_y = obj_data[0], obj_data[1]
            obj_type = obj_data[3]
            obj_yaw = obj_data[5]

            dx = obj_x - ref_x
            dy = obj_y - ref_y

            rel_x = dx * math.cos(-ref_yaw_rad) - dy * math.sin(-ref_yaw_rad)
            rel_y = dx * math.sin(-ref_yaw_rad) + dy * math.cos(-ref_yaw_rad)

            rel_yaw = obj_yaw - ref_yaw
            while rel_yaw > 180:
                rel_yaw -= 360
            while rel_yaw < -180:
                rel_yaw += 360

            result[obj_name] = {
                "x": rel_x,
                "y": rel_y,
                "z": self.DEFAULT_Z.get(obj_name, 0.0),
                "yaw": rel_yaw,
            }

        return result

    def log_request_positions(self, selected_frame, request_positions):
        if "reference" in self.placed_objects:
            ref_data = self.placed_objects["reference"]
            rospy.loginfo(
                "[PremapGUI] Reference absolute in pool-map: "
                f"x={ref_data[0]:.3f}, y={ref_data[1]:.3f}, yaw={ref_data[5]:.1f}"
            )
        else:
            rospy.loginfo("[PremapGUI] Reference absolute in pool-map: <not placed>")

        rospy.loginfo(
            f"[PremapGUI] Request poses in '{selected_frame}' frame "
            f"(count={len(request_positions)}):"
        )

        for obj_name in sorted(request_positions.keys()):
            pose_data = request_positions[obj_name]
            service_labels = self.SERVICE_LABEL_MAP.get(obj_name, [obj_name])
            rospy.loginfo(
                f"[PremapGUI]   {obj_name}: x={pose_data['x']:.3f}, "
                f"y={pose_data['y']:.3f}, z={pose_data['z']:.3f}, "
                f"yaw={pose_data['yaw']:.1f}, labels={service_labels}"
            )

    def send_to_vehicle(self):
        selected_frame = self.reference_frame_var.get()

        if "reference" not in self.placed_objects:
            messagebox.showerror("Error", "Place reference frame first!")
            return

        relative_positions = self.get_relative_positions()

        if not relative_positions:
            messagebox.showwarning("Warning", "No objects placed (except reference)")
            return

        if not self.service_connected:
            messagebox.showerror(
                "Error", "ROS service not connected!\nMake sure ROS master is running."
            )
            return

        try:
            req = SetPremapRequest()
            req.reference_frame = selected_frame
            req.objects = []

            sent_count = 0

            for obj_name, pos_data in relative_positions.items():
                service_labels = self.SERVICE_LABEL_MAP.get(obj_name, [obj_name])
                for service_label in service_labels:
                    obj_pose = ObjectPose()
                    obj_pose.label = service_label

                    obj_pose.pose.position.x = pos_data["x"]
                    obj_pose.pose.position.y = pos_data["y"]
                    obj_pose.pose.position.z = pos_data["z"]

                    yaw_rad = math.radians(pos_data["yaw"])
                    q = quaternion_from_euler(0, 0, yaw_rad)
                    obj_pose.pose.orientation.x = q[0]
                    obj_pose.pose.orientation.y = q[1]
                    obj_pose.pose.orientation.z = q[2]
                    obj_pose.pose.orientation.w = q[3]

                    req.objects.append(obj_pose)
                    sent_count += 1

            if sent_count == 0:
                messagebox.showwarning("Warning", "No objects placed to send")
                return

            self.log_request_positions(selected_frame, relative_positions)

            rospy.loginfo(
                f"[PremapGUI] Sending SetPremap: reference_frame='{req.reference_frame}', "
                f"objects={[obj.label for obj in req.objects]}"
            )

            rospy.wait_for_service(self.service_name, timeout=2.0)
            set_premap = rospy.ServiceProxy(self.service_name, SetPremap)
            resp = set_premap(req)

            rospy.loginfo(
                f"[PremapGUI] SetPremap response: success={resp.success}, "
                f"message='{resp.message}'"
            )

            if resp.success:
                messagebox.showinfo(
                    "Success", f"Sent {sent_count} objects to vehicle!\n{resp.message}"
                )
            else:
                messagebox.showerror(
                    "Error", f"Service returned error:\n{resp.message}"
                )

        except rospy.ServiceException as e:
            messagebox.showerror("Error", f"Service call failed:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send:\n{e}")


def main():
    rospy.init_node("competition_map_gui", anonymous=True)

    root = tk.Tk()
    CompetitionMapGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

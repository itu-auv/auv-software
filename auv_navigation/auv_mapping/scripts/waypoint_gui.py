#!/usr/bin/env python3

import math
import signal
import tkinter as tk
from tkinter import messagebox, ttk

import rospy
import yaml
from auv_msgs.msg import WaypointPath
from auv_msgs.srv import SetWaypoint, SetWaypointRequest
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, TriggerRequest
from tf.transformations import quaternion_from_euler


PATH_COLORS = [
    "#9c27b0",
    "#2196F3",
    "#ff9800",
    "#4caf50",
    "#e91e63",
    "#00bcd4",
    "#795548",
    "#607d8b",
]

B_MODE_FIXED = 0
B_MODE_RELATIVE = 1


class PathData:
    def __init__(
        self,
        index,
        color,
        waypoint_prefix,
        default_z,
        default_ref_a,
        default_ref_b,
        default_b_mode,
        b_reference_distance,
    ):
        self.index = index
        self.name = f"path{index}"
        self.color = color
        self.waypoint_prefix = waypoint_prefix

        self.ref_a = tk.StringVar(value=default_ref_a)
        self.ref_b = tk.StringVar(value=default_ref_b)
        self.b_mode = tk.IntVar(value=default_b_mode)
        self.b_reference_distance = float(b_reference_distance)
        self.yaw_var = tk.DoubleVar(value=0.0)
        self.z_var = tk.DoubleVar(value=default_z)

        self.waypoints = []
        self.selected_index = None

    def wp_frame_name(self, i):
        return f"{self.waypoint_prefix}{self.index}_wp{i + 1}"


class WaypointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waypoint GUI")
        self.root.attributes("-zoomed", True)
        self._is_closing = False

        self.service_name = rospy.get_param("~set_waypoint_service", "set_waypoint")
        self.get_service_name = rospy.get_param("~get_waypoint_service", "get_waypoint")
        self.reference_frame_options = list(
            rospy.get_param(
                "~reference_frame_options",
                ["odom", "coin_flip"],
            )
        )
        self.waypoint_prefix = rospy.get_param("~waypoint_prefix", "path")
        self.default_z = float(rospy.get_param("~default_waypoint_z", -1.0))

        self.x_min = float(rospy.get_param("~canvas_x_min", -8.0))
        self.x_max = float(rospy.get_param("~canvas_x_max", 8.0))
        self.y_min = float(rospy.get_param("~canvas_y_min", -2.0))
        self.y_max = float(rospy.get_param("~canvas_y_max", 18.0))
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Invalid canvas bounds")
        self.pool_width = self.x_max - self.x_min
        self.pool_height = self.y_max - self.y_min
        self.b_arrow_length = float(rospy.get_param("~b_arrow_length", 12.0))
        self.default_b_mode = self._normalize_b_mode(
            rospy.get_param("~default_b_mode", B_MODE_FIXED)
        )

        self.paths = []
        self.active_path_idx = None
        self.service_connected = False

        self.canvas_padding = 40
        self.scale = 1.0
        self.pool_x_offset = 0
        self.pool_y_offset = 0

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind_all("<Control-c>", lambda _event: self.close())
        self.root.bind_all("<Control-C>", lambda _event: self.close())
        self.add_path()
        self.root.after_idle(self.redraw_all)
        self.root.after(100, self.redraw_all)
        self.root.after(100, self._poll_signal_events)
        self.root.after(500, self.check_ros_service)

    def _poll_signal_events(self):
        if not self._is_closing:
            self.root.after(100, self._poll_signal_events)

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Waypoint GUI closed")
        self.root.quit()
        self.root.destroy()

    def check_ros_service(self):
        was_connected = self.service_connected
        try:
            rospy.wait_for_service(self.service_name, timeout=1.0)
            self.service_connected = True
            self.status_label.config(text="ROS: Connected", fg="green")
            if not was_connected:
                rospy.loginfo(f"[WaypointGUI] Service connected: {self.service_name}")
        except rospy.ROSException:
            self.service_connected = False
            self.status_label.config(text="ROS: Not connected", fg="red")
            if was_connected:
                rospy.logwarn(
                    f"[WaypointGUI] Service disconnected: {self.service_name}"
                )

        self.root.after(5000, self.check_ros_service)

    def setup_ui(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, width=360)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        top_row = tk.Frame(left_frame)
        top_row.pack(fill=tk.X, pady=(0, 5))
        tk.Button(
            top_row,
            text="+ New Path",
            command=self.add_path,
            bg="#4caf50",
            fg="white",
            font=("Arial", 9, "bold"),
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            top_row,
            text="- Remove Path",
            command=self.remove_active_path,
            bg="#f44336",
            fg="white",
        ).pack(side=tk.LEFT, padx=2)

        tk.Button(
            left_frame,
            text="Send to Vehicle",
            command=self.send_to_vehicle,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(fill=tk.X, pady=(0, 5))

        tk.Button(
            left_frame,
            text="Get from Vehicle",
            command=self.get_from_vehicle,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
        ).pack(fill=tk.X, pady=(0, 5))

        self.status_label = tk.Label(left_frame, text="ROS: Checking...", fg="orange")
        self.status_label.pack(fill=tk.X, pady=(0, 5))

        self.notebook = ttk.Notebook(left_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

        self.coord_label = tk.Label(left_frame, text="Mouse: x: -  y: -")
        self.coord_label.pack(pady=(5, 2))

        right_frame = tk.LabelFrame(
            main_frame,
            text=(
                f"Per-path local frame  "
                f"x:[{self.x_min:.1f}, {self.x_max:.1f}]  "
                f"y:[{self.y_min:.1f}, {self.y_max:.1f}]   "
                f"(A = origin, +x = A to B)"
            ),
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(right_frame, bg="#e6f2ff")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Configure>", self.on_resize)

    def _build_tab(self, path):
        tab = tk.Frame(self.notebook)
        path._tab = tab
        path._controls = {}

        ref_box = tk.LabelFrame(
            tab,
            text="Reference frame (A to B composite, or A only if B is empty)",
        )
        ref_box.pack(fill=tk.X, padx=5, pady=5)

        a_row = tk.Frame(ref_box)
        a_row.pack(fill=tk.X, padx=5, pady=3)
        tk.Label(a_row, text="A (origin):", width=14, anchor="w").pack(side=tk.LEFT)
        a_combo = ttk.Combobox(
            a_row,
            textvariable=path.ref_a,
            values=self.reference_frame_options,
            width=22,
        )
        a_combo.pack(side=tk.LEFT, padx=(5, 0))
        a_combo.bind("<<ComboboxSelected>>", lambda _e: self.redraw_dynamic())
        path.ref_a.trace_add("write", lambda *_args: self.redraw_dynamic())

        b_row = tk.Frame(ref_box)
        b_row.pack(fill=tk.X, padx=5, pady=3)
        tk.Label(b_row, text="B (+x direction):", width=14, anchor="w").pack(
            side=tk.LEFT
        )
        b_combo = ttk.Combobox(
            b_row,
            textvariable=path.ref_b,
            values=self.reference_frame_options,
            width=22,
        )
        b_combo.pack(side=tk.LEFT, padx=(5, 0))
        b_combo.bind("<<ComboboxSelected>>", lambda _e: self.redraw_dynamic())
        path.ref_b.trace_add("write", lambda *_args: self.redraw_dynamic())
        path.b_mode.trace_add("write", lambda *_args: self.redraw_all())

        mode_box = tk.LabelFrame(tab, text="B mode")
        mode_box.pack(fill=tk.X, padx=5, pady=5)
        tk.Radiobutton(
            mode_box,
            text="Use GUI metres (+x only)",
            variable=path.b_mode,
            value=B_MODE_FIXED,
        ).pack(anchor="w", padx=5, pady=1)
        tk.Radiobutton(
            mode_box,
            text="Scale by A-B distance",
            variable=path.b_mode,
            value=B_MODE_RELATIVE,
        ).pack(anchor="w", padx=5, pady=1)

        wp_box = tk.LabelFrame(tab, text="New / selected waypoint")
        wp_box.pack(fill=tk.X, padx=5, pady=5)
        tk.Scale(
            wp_box,
            from_=180,
            to=-180,
            orient=tk.HORIZONTAL,
            variable=path.yaw_var,
            label="Yaw (deg, in A to B frame)",
            command=lambda v, p=path: self._on_yaw_change(p, v),
        ).pack(fill=tk.X, padx=5, pady=2)
        z_row = tk.Frame(wp_box)
        z_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(z_row, text="Z (m, in A to B frame):").pack(side=tk.LEFT)
        tk.Entry(z_row, textvariable=path.z_var, width=8).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        btn_row = tk.Frame(tab)
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(btn_row, text="Undo", command=lambda p=path: self._undo(p)).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2)
        )
        tk.Button(
            btn_row,
            text="Delete Sel.",
            command=lambda p=path: self._delete_selected(p),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(
            btn_row,
            text="Clear",
            command=lambda p=path: self._clear_all(p),
            bg="#f44336",
            fg="white",
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        tk.Label(
            tab,
            text="Waypoints (click to select):",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w", padx=5, pady=(5, 2))
        list_frame = tk.Frame(tab)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        listbox = tk.Listbox(list_frame, font=("Courier", 9), activestyle="dotbox")
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=sb.set)
        listbox.bind("<<ListboxSelect>>", lambda _e, p=path: self._on_list_select(p))
        path._controls["listbox"] = listbox

        return tab

    def add_path(self):
        idx = len(self.paths) + 1
        color = PATH_COLORS[(idx - 1) % len(PATH_COLORS)]
        opts = self.reference_frame_options
        default_a = opts[0] if opts else ""
        default_b = opts[1] if len(opts) > 1 else ""
        path = PathData(
            index=idx,
            color=color,
            waypoint_prefix=self.waypoint_prefix,
            default_z=self.default_z,
            default_ref_a=default_a,
            default_ref_b=default_b,
            default_b_mode=self.default_b_mode,
            b_reference_distance=self.b_arrow_length,
        )
        tab = self._build_tab(path)
        self.paths.append(path)
        self.notebook.add(tab, text=path.name)
        self.notebook.select(tab)
        self.active_path_idx = len(self.paths) - 1
        self._refresh_listbox(path)
        self.redraw_all()

    def remove_active_path(self):
        if self.active_path_idx is None or not self.paths:
            return
        if len(self.paths) == 1:
            messagebox.showinfo("Cannot remove", "At least one path must remain.")
            return
        idx = self.active_path_idx
        path = self.paths.pop(idx)
        self.notebook.forget(path._tab)
        for i, p in enumerate(self.paths, start=1):
            p.index = i
            p.name = f"path{i}"
            self.notebook.tab(p._tab, text=p.name)
            self._refresh_listbox(p)
        self.active_path_idx = min(idx, len(self.paths) - 1)
        self.redraw_all()

    def clear_paths(self):
        for path in self.paths:
            self.notebook.forget(path._tab)
        self.paths = []
        self.active_path_idx = None

    def _on_tab_change(self, _event):
        current = self.notebook.select()
        if not current:
            return
        for i, p in enumerate(self.paths):
            if str(p._tab) == current:
                self.active_path_idx = i
                break
        self.redraw_all()

    def _active_path(self):
        if self.active_path_idx is None:
            return None
        if not (0 <= self.active_path_idx < len(self.paths)):
            return None
        return self.paths[self.active_path_idx]

    def on_resize(self, _event):
        self.redraw_all()

    def redraw_all(self):
        self.draw_pool()
        self.redraw_dynamic()

    def _is_scale_mode(self):
        path = self._active_path()
        return path is not None and path.b_mode.get() == B_MODE_RELATIVE

    def draw_pool(self):
        self.canvas.delete("pool")
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100 or canvas_h < 100:
            return

        available_w = canvas_w - 2 * self.canvas_padding
        available_h = canvas_h - 2 * self.canvas_padding
        self.scale = min(available_w / self.pool_width, available_h / self.pool_height)

        pool_w = self.pool_width * self.scale
        pool_h = self.pool_height * self.scale
        self.pool_x_offset = (canvas_w - pool_w) / 2
        self.pool_y_offset = (canvas_h - pool_h) / 2

        self.canvas.create_rectangle(
            self.pool_x_offset,
            self.pool_y_offset,
            self.pool_x_offset + pool_w,
            self.pool_y_offset + pool_h,
            outline="#1a237e",
            width=3,
            fill="white",
            tags="pool",
        )

        grid_step = 2
        y_val = math.ceil(self.y_min / grid_step) * grid_step
        while y_val <= self.y_max + 1e-6:
            py = self.pool_y_offset + pool_h - (y_val - self.y_min) * self.scale
            is_axis = abs(y_val) < 1e-6
            self.canvas.create_line(
                self.pool_x_offset,
                py,
                self.pool_x_offset + pool_w,
                py,
                fill="#ef5350" if is_axis else "#b0bec5",
                dash=() if is_axis else (2, 4),
                tags="pool",
            )
            if not self._is_scale_mode():
                self.canvas.create_text(
                    self.pool_x_offset - 18,
                    py,
                    text=f"{y_val:.0f}",
                    font=("Arial", 8),
                    fill="#b71c1c" if is_axis else "black",
                    tags="pool",
                )
            y_val += grid_step

        x_val = math.ceil(self.x_min / grid_step) * grid_step
        while x_val <= self.x_max + 1e-6:
            px = self.pool_x_offset + (x_val - self.x_min) * self.scale
            is_axis = abs(x_val) < 1e-6
            self.canvas.create_line(
                px,
                self.pool_y_offset,
                px,
                self.pool_y_offset + pool_h,
                fill="#ef5350" if is_axis else "#b0bec5",
                dash=() if is_axis else (2, 4),
                tags="pool",
            )
            if not self._is_scale_mode():
                self.canvas.create_text(
                    px,
                    self.pool_y_offset + pool_h + 12,
                    text=f"{x_val:.0f}",
                    font=("Arial", 8),
                    fill="#b71c1c" if is_axis else "black",
                    tags="pool",
                )
            x_val += grid_step

    def ref_to_pixels(self, x_ref, y_ref):
        canvas_x_m = -y_ref
        canvas_y_m = x_ref
        pool_h = self.pool_height * self.scale
        px = self.pool_x_offset + (canvas_x_m - self.x_min) * self.scale
        py = self.pool_y_offset + pool_h - (canvas_y_m - self.y_min) * self.scale
        return px, py

    def pixels_to_ref(self, px, py):
        pool_h = self.pool_height * self.scale
        canvas_x_m = (px - self.pool_x_offset) / self.scale + self.x_min
        canvas_y_m = (self.pool_y_offset + pool_h - py) / self.scale + self.y_min
        return canvas_y_m, -canvas_x_m

    def _in_bounds_ref(self, x_ref, y_ref):
        canvas_x = -y_ref
        canvas_y = x_ref
        return (
            self.x_min <= canvas_x <= self.x_max
            and self.y_min <= canvas_y <= self.y_max
        )

    def on_mouse_move(self, event):
        x_ref, y_ref = self.pixels_to_ref(event.x, event.y)
        if self._in_bounds_ref(x_ref, y_ref):
            self.coord_label.config(text=f"Mouse: x: {x_ref:+.2f}m  y: {y_ref:+.2f}m")
        else:
            self.coord_label.config(text="Mouse: x: -  y: -")

    def on_click(self, event):
        path = self._active_path()
        if path is None:
            return
        x_ref, y_ref = self.pixels_to_ref(event.x, event.y)
        if not self._in_bounds_ref(x_ref, y_ref):
            return

        try:
            z = float(path.z_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Invalid Z", "Waypoint Z must be a number.")
            return

        path.waypoints.append(
            {
                "x": x_ref,
                "y": y_ref,
                "z": z,
                "yaw": float(path.yaw_var.get()),
            }
        )
        path.selected_index = len(path.waypoints) - 1
        self._refresh_listbox(path)
        self.redraw_dynamic()

    def _on_list_select(self, path):
        lb = path._controls["listbox"]
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        path.selected_index = idx
        wp = path.waypoints[idx]
        path.yaw_var.set(wp["yaw"])
        path.z_var.set(wp["z"])
        self.redraw_dynamic()

    def _on_yaw_change(self, path, value):
        if path.selected_index is None:
            return
        if 0 <= path.selected_index < len(path.waypoints):
            path.waypoints[path.selected_index]["yaw"] = float(value)
            self._refresh_listbox(path)
            self.redraw_dynamic()

    def _undo(self, path):
        if not path.waypoints:
            return
        path.waypoints.pop()
        if path.selected_index is not None and path.selected_index >= len(
            path.waypoints
        ):
            path.selected_index = len(path.waypoints) - 1 if path.waypoints else None
        self._refresh_listbox(path)
        self.redraw_dynamic()

    def _delete_selected(self, path):
        if path.selected_index is None:
            return
        if 0 <= path.selected_index < len(path.waypoints):
            path.waypoints.pop(path.selected_index)
            if not path.waypoints:
                path.selected_index = None
            elif path.selected_index >= len(path.waypoints):
                path.selected_index = len(path.waypoints) - 1
            self._refresh_listbox(path)
            self.redraw_dynamic()

    def _clear_all(self, path):
        path.waypoints.clear()
        path.selected_index = None
        self._refresh_listbox(path)
        self.redraw_dynamic()

    def _refresh_listbox(self, path):
        lb = path._controls["listbox"]
        lb.delete(0, tk.END)
        for i, wp in enumerate(path.waypoints):
            name = path.wp_frame_name(i)
            lb.insert(
                tk.END,
                f"{name:<14} x={wp['x']:+6.2f} y={wp['y']:+6.2f} "
                f"z={wp['z']:+5.2f} yaw={wp['yaw']:+6.1f}",
            )
        if path.selected_index is not None and 0 <= path.selected_index < len(
            path.waypoints
        ):
            lb.selection_clear(0, tk.END)
            lb.selection_set(path.selected_index)
            lb.activate(path.selected_index)

    def redraw_dynamic(self):
        self.canvas.delete("dynamic")
        path = self._active_path()
        if path is None:
            return
        self._draw_origin_and_vector(path)
        self._draw_waypoints(path)

    def _draw_origin_and_vector(self, path):
        a_name = path.ref_a.get() or "A"
        b_name = path.ref_b.get() or "B"
        is_scale_mode = path.b_mode.get() == B_MODE_RELATIVE

        ox, oy = self.ref_to_pixels(0.0, 0.0)
        r = 8
        self.canvas.create_oval(
            ox - r,
            oy - r,
            ox + r,
            oy + r,
            fill="#ffeb3b",
            outline="#f57f17",
            width=2,
            tags="dynamic",
        )
        self.canvas.create_text(
            ox + 14,
            oy + 4,
            text=a_name,
            anchor="w",
            font=("Arial", 10, "bold"),
            fill="#b26a00",
            tags="dynamic",
        )

        bx, by = self.ref_to_pixels(self.b_arrow_length, 0.0)
        self.canvas.create_line(
            ox,
            oy,
            bx,
            by,
            fill="#5d4037",
            width=2,
            dash=(8, 4),
            arrow=tk.NONE if is_scale_mode else tk.LAST,
            tags="dynamic",
        )
        if is_scale_mode:
            br = 8
            self.canvas.create_oval(
                bx - br,
                by - br,
                bx + br,
                by + br,
                fill="#80deea",
                outline="#00838f",
                width=2,
                tags="dynamic",
            )
        self.canvas.create_text(
            bx,
            by - 14,
            text=self._b_label(path, b_name),
            font=("Arial", 10, "bold"),
            fill="#5d4037",
            tags="dynamic",
        )

    def _b_label(self, path, b_name):
        if path.b_mode.get() == B_MODE_RELATIVE:
            return f"{b_name}  (relative +x)"
        return f"{b_name}  (+x)"

    def _draw_waypoints(self, path):
        pts = []
        for wp in path.waypoints:
            px, py = self.ref_to_pixels(wp["x"], wp["y"])
            pts.append((px, py, wp["yaw"]))

        for i in range(len(pts) - 1):
            x1, y1, _ = pts[i]
            x2, y2, _ = pts[i + 1]
            self.canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=path.color,
                width=2,
                dash=(4, 2),
                arrow=tk.LAST,
                tags="dynamic",
            )

        for i, (px, py, yaw_deg) in enumerate(pts):
            is_selected = i == path.selected_index
            r = 9 if is_selected else 7
            outline = "#ff6d00" if is_selected else path.color
            self.canvas.create_oval(
                px - r,
                py - r,
                px + r,
                py + r,
                fill="#ffffff",
                outline=outline,
                width=2,
                tags="dynamic",
            )
            arrow_len = 1.0 * self.scale
            screen_angle = math.radians(yaw_deg + 90.0)
            dx = arrow_len * math.cos(screen_angle)
            dy = arrow_len * math.sin(screen_angle)
            self.canvas.create_line(
                px,
                py,
                px + dx,
                py - dy,
                fill=outline,
                width=2,
                arrow=tk.LAST,
                tags="dynamic",
            )
            self.canvas.create_text(
                px,
                py + 14,
                text=f"{path.name}.wp{i + 1}",
                font=("Arial", 9, "bold"),
                fill=outline,
                tags="dynamic",
            )

    def _build_service_request(self):
        req = SetWaypointRequest()
        req.paths = [self._path_to_msg(path) for path in self.paths if path.waypoints]
        return req

    def _path_to_msg(self, path):
        msg = WaypointPath()
        msg.name = path.name
        msg.ref_a = path.ref_a.get()
        msg.ref_b = path.ref_b.get()
        msg.b_mode = self._normalize_b_mode(path.b_mode.get())
        msg.b_reference_distance = float(path.b_reference_distance)
        msg.waypoint_prefix = path.waypoint_prefix
        msg.waypoints = [self._waypoint_to_pose(wp) for wp in path.waypoints]
        return msg

    def _normalize_b_mode(self, value):
        try:
            mode = int(value)
        except (TypeError, ValueError):
            return B_MODE_FIXED
        return mode if mode in (B_MODE_FIXED, B_MODE_RELATIVE) else B_MODE_FIXED

    def _waypoint_to_pose(self, wp):
        pose = Pose()
        pose.position.x = float(wp["x"])
        pose.position.y = float(wp["y"])
        pose.position.z = float(wp["z"])
        q = quaternion_from_euler(0.0, 0.0, math.radians(float(wp["yaw"])))
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose

    def get_from_vehicle(self):
        try:
            rospy.wait_for_service(self.get_service_name, timeout=2.0)
            get_waypoint = rospy.ServiceProxy(self.get_service_name, Trigger)
            resp = get_waypoint(TriggerRequest())

            if not resp.success:
                messagebox.showerror("Error", resp.message)
                return

            state = yaml.safe_load(resp.message) or {}
            paths = state.get("paths", [])
            if not paths:
                messagebox.showwarning("Warning", "Vehicle has no waypoint paths.")
                return

            self.apply_vehicle_paths(paths)
            waypoint_count = sum(len(path.waypoints) for path in self.paths)
            messagebox.showinfo(
                "Success",
                f"Loaded {len(self.paths)} path(s), {waypoint_count} waypoint(s).",
            )
        except rospy.ServiceException as exc:
            messagebox.showerror("Error", f"Service call failed:\n{exc}")
        except rospy.ROSException as exc:
            messagebox.showerror("Error", f"Service unavailable:\n{exc}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to get waypoints:\n{exc}")

    def apply_vehicle_paths(self, path_entries):
        self.clear_paths()

        for entry in path_entries:
            path = self._path_from_state_dict(entry)
            if path is None:
                continue
            tab = self._build_tab(path)
            self.paths.append(path)
            self.notebook.add(tab, text=path.name)
            self._refresh_listbox(path)

        if not self.paths:
            self.add_path()
            raise ValueError("No valid waypoint paths in vehicle response")

        self.active_path_idx = 0
        self.notebook.select(self.paths[0]._tab)
        self.redraw_all()

    def _path_from_state_dict(self, entry):
        if not isinstance(entry, dict):
            return None

        idx = len(self.paths) + 1
        color = PATH_COLORS[(idx - 1) % len(PATH_COLORS)]
        path = PathData(
            index=idx,
            color=color,
            waypoint_prefix=str(entry.get("waypoint_prefix") or self.waypoint_prefix),
            default_z=self.default_z,
            default_ref_a=str(entry.get("ref_a") or ""),
            default_ref_b=str(entry.get("ref_b") or ""),
            default_b_mode=self._normalize_b_mode(entry.get("b_mode", B_MODE_FIXED)),
            b_reference_distance=float(
                entry.get("b_reference_distance") or self.b_arrow_length
            ),
        )
        path.name = str(entry.get("name") or f"path{idx}")

        waypoints = []
        for wp in entry.get("waypoints", []) or []:
            try:
                waypoints.append(
                    {
                        "x": float(wp["x"]),
                        "y": float(wp["y"]),
                        "z": float(wp.get("z", self.default_z)),
                        "yaw": float(wp.get("yaw", 0.0)),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        if not waypoints:
            return None

        path.waypoints = waypoints
        path.selected_index = 0
        path.yaw_var.set(waypoints[0]["yaw"])
        path.z_var.set(waypoints[0]["z"])
        return path

    def send_to_vehicle(self):
        if not self.service_connected:
            messagebox.showerror(
                "Error", "ROS service not connected. Start waypoint_publisher first."
            )
            return

        waypoint_count = sum(len(path.waypoints) for path in self.paths)
        if waypoint_count == 0:
            messagebox.showwarning("Warning", "No waypoints to send.")
            return

        try:
            rospy.wait_for_service(self.service_name, timeout=2.0)
            set_waypoint = rospy.ServiceProxy(self.service_name, SetWaypoint)
            resp = set_waypoint(self._build_service_request())
            if resp.success:
                messagebox.showinfo("Success", resp.message)
            else:
                messagebox.showerror("Error", resp.message)
        except rospy.ServiceException as exc:
            messagebox.showerror("Error", f"Service call failed:\n{exc}")
        except rospy.ROSException as exc:
            messagebox.showerror("Error", f"Service unavailable:\n{exc}")


def main():
    rospy.init_node("waypoint_gui", anonymous=True)
    root = tk.Tk()
    gui = WaypointGUI(root)
    signal.signal(signal.SIGINT, lambda _sig, _frame: root.after(0, gui.close))
    root.mainloop()


if __name__ == "__main__":
    main()

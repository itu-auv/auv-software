#!/usr/bin/env python3

import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
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


class PathData:
    """A single editable path: two frames (A, B) define a composite ref; waypoints
    are stored in that ref frame with +x = A→B."""

    def __init__(
        self, index, color, waypoint_prefix, default_z, default_ref_a, default_ref_b
    ):
        self.index = index  # 1-based
        self.name = f"path{index}"
        self.color = color
        self.waypoint_prefix = waypoint_prefix

        self.ref_a = tk.StringVar(value=default_ref_a)
        self.ref_b = tk.StringVar(value=default_ref_b)
        self.publish_enabled = tk.BooleanVar(value=True)

        self.yaw_var = tk.DoubleVar(value=0.0)
        self.z_var = tk.DoubleVar(value=default_z)

        self.waypoints = []  # [{x, y, z, yaw}] in composite ref coords
        self.selected_index = None

    def composite_frame_name(self):
        return f"{self.name}_ref"

    def wp_frame_name(self, i):
        return f"{self.waypoint_prefix}{self.index}_wp{i + 1}"


class WaypointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waypoint GUI (multi-path, composite refs)")
        self.root.attributes("-zoomed", True)

        self.world_frame = rospy.get_param("~world_frame", "odom")
        self.known_object_frames = list(rospy.get_param("~known_object_frames", []))
        default_options = ["coin_flip"] + self.known_object_frames
        self.reference_frame_options = list(
            rospy.get_param(
                "~reference_frame_options",
                default_options,
            )
        )
        self.waypoint_prefix = rospy.get_param("~waypoint_prefix", "path")
        self.default_z = float(rospy.get_param("~default_waypoint_z", -1.0))
        self.broadcast_rate_hz = float(rospy.get_param("~broadcast_rate_hz", 20.0))

        self.x_min = float(rospy.get_param("~canvas_x_min", -8.0))
        self.x_max = float(rospy.get_param("~canvas_x_max", 8.0))
        self.y_min = float(rospy.get_param("~canvas_y_min", -2.0))
        self.y_max = float(rospy.get_param("~canvas_y_max", 18.0))
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Invalid canvas bounds")
        self.pool_width = self.x_max - self.x_min
        self.pool_height = self.y_max - self.y_min

        self.b_arrow_length = float(rospy.get_param("~b_arrow_length", 14.0))
        self.paths_rosparam = rospy.get_param("~paths_rosparam", "/waypoint_gui/paths")

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        self.simulate_var = tk.BooleanVar(value=False)
        self._rosparam_sync_counter = 0

        self.paths = []
        self.active_path_idx = None

        self.canvas_padding = 40
        self.scale = 1.0
        self.pool_x_offset = 0
        self.pool_y_offset = 0

        self._paths_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        self.setup_ui()
        self.add_path()

        self._broadcast_thread = threading.Thread(
            target=self._broadcast_loop, daemon=True
        )
        self._broadcast_thread.start()

        self.root.after(100, self.draw_pool)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
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

        sim_row = tk.Frame(left_frame)
        sim_row.pack(fill=tk.X, pady=(0, 4))
        tk.Checkbutton(
            sim_row,
            text="Simulate missing A/B frames (offline test)",
            variable=self.simulate_var,
            font=("Arial", 9),
        ).pack(side=tk.LEFT, padx=2)

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
                f"(A = origin, +x = A→B)"
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

        path._status_label = tk.Label(
            tab,
            text="○ initialising…",
            fg="gray",
            font=("Arial", 9, "italic"),
            anchor="w",
            justify="left",
            wraplength=340,
        )
        path._status_label.pack(fill=tk.X, padx=5, pady=(2, 2))

        ref_box = tk.LabelFrame(tab, text="Reference frame (A → B composite)")
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

        tk.Checkbutton(
            ref_box,
            text="Publish this path's frames (TF broadcast)",
            variable=path.publish_enabled,
        ).pack(anchor="w", padx=5, pady=(4, 5))

        wp_box = tk.LabelFrame(tab, text="New / selected waypoint")
        wp_box.pack(fill=tk.X, padx=5, pady=5)
        tk.Scale(
            wp_box,
            from_=180,
            to=-180,
            orient=tk.HORIZONTAL,
            variable=path.yaw_var,
            label="Yaw (deg, in A→B frame)",
            command=lambda v, p=path: self._on_yaw_change(p, v),
        ).pack(fill=tk.X, padx=5, pady=2)
        z_row = tk.Frame(wp_box)
        z_row.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(z_row, text="Z (m, in A→B frame):").pack(side=tk.LEFT)
        tk.Entry(z_row, textvariable=path.z_var, width=8).pack(
            side=tk.LEFT, padx=(5, 0)
        )

        btn_row = tk.Frame(tab)
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        tk.Button(
            btn_row,
            text="Undo",
            command=lambda p=path: self._undo(p),
        ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
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
        listbox.bind(
            "<<ListboxSelect>>",
            lambda _e, p=path: self._on_list_select(p),
        )
        path._controls["listbox"] = listbox

        return tab

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------
    def add_path(self):
        idx = len(self.paths) + 1
        color = PATH_COLORS[(idx - 1) % len(PATH_COLORS)]
        opts = self.reference_frame_options
        default_a = opts[0] if opts else ""
        default_b = opts[1] if len(opts) > 1 else default_a
        path = PathData(
            index=idx,
            color=color,
            waypoint_prefix=self.waypoint_prefix,
            default_z=self.default_z,
            default_ref_a=default_a,
            default_ref_b=default_b,
        )
        tab = self._build_tab(path)
        with self._paths_lock:
            self.paths.append(path)
        self.notebook.add(tab, text=path.name)
        self.notebook.select(tab)
        self.active_path_idx = len(self.paths) - 1
        self._refresh_listbox(path)
        self.redraw_dynamic()

    def remove_active_path(self):
        if self.active_path_idx is None or not self.paths:
            return
        if len(self.paths) == 1:
            messagebox.showinfo("Cannot remove", "At least one path must remain.")
            return
        idx = self.active_path_idx
        with self._paths_lock:
            path = self.paths.pop(idx)
        self.notebook.forget(path._tab)
        for i, p in enumerate(self.paths, start=1):
            p.index = i
            p.name = f"path{i}"
            self.notebook.tab(p._tab, text=p.name)
            self._refresh_listbox(p)
        self.active_path_idx = min(idx, len(self.paths) - 1)
        self.redraw_dynamic()

    def _on_tab_change(self, _event):
        current = self.notebook.select()
        if not current:
            return
        for i, p in enumerate(self.paths):
            if str(p._tab) == current:
                self.active_path_idx = i
                break
        self.redraw_dynamic()

    def _active_path(self):
        if self.active_path_idx is None or not self.paths:
            return None
        if not (0 <= self.active_path_idx < len(self.paths)):
            return None
        return self.paths[self.active_path_idx]

    # ------------------------------------------------------------------
    # Canvas drawing
    # ------------------------------------------------------------------
    def on_resize(self, _event):
        self.draw_pool()
        self.redraw_dynamic()

    def draw_pool(self):
        self.canvas.delete("pool")
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 100 or canvas_h < 100:
            return

        available_w = canvas_w - 2 * self.canvas_padding
        available_h = canvas_h - 2 * self.canvas_padding
        self.scale = min(
            available_w / self.pool_width,
            available_h / self.pool_height,
        )

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
        """Ref-frame coords (x forward, y left) → canvas pixels (x up, y left).

        We rotate the ref frame so that its +x (forward) appears upward on the
        canvas and its +y (left) appears leftward. That means:
            canvas_horizontal = -y_ref   (right = negative y_ref = ref's +y goes left on screen → invert for display)
            canvas_vertical   = +x_ref   (upward)
        With canvas_horizontal mapped to screen x through x_min..x_max and
        canvas_vertical mapped to screen y (inverted).
        """
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
        x_ref = canvas_y_m
        y_ref = -canvas_x_m
        return x_ref, y_ref

    def _in_bounds_ref(self, x_ref, y_ref):
        canvas_x = -y_ref
        canvas_y = x_ref
        return (
            self.x_min <= canvas_x <= self.x_max
            and self.y_min <= canvas_y <= self.y_max
        )

    # ------------------------------------------------------------------
    # Mouse / list handlers
    # ------------------------------------------------------------------
    def on_mouse_move(self, event):
        x_ref, y_ref = self.pixels_to_ref(event.x, event.y)
        if self._in_bounds_ref(x_ref, y_ref):
            self.coord_label.config(
                text=f"Mouse: x: {x_ref:+.2f}m  y: {y_ref:+.2f}m  (in A→B frame)"
            )
        else:
            self.coord_label.config(text="Mouse: x: -  y: -")

    def on_click(self, event):
        path = self._active_path()
        if path is None:
            return
        x_ref, y_ref = self.pixels_to_ref(event.x, event.y)
        if not self._in_bounds_ref(x_ref, y_ref):
            return

        path.waypoints.append(
            {
                "x": x_ref,
                "y": y_ref,
                "z": float(path.z_var.get()),
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
                f"{name:<14} x={wp['x']:+6.2f} y={wp['y']:+6.2f} yaw={wp['yaw']:+6.1f}",
            )
        if path.selected_index is not None and 0 <= path.selected_index < len(
            path.waypoints
        ):
            lb.selection_clear(0, tk.END)
            lb.selection_set(path.selected_index)
            lb.activate(path.selected_index)

    # ------------------------------------------------------------------
    # Drawing overlay (only the active path's local frame)
    # ------------------------------------------------------------------
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

        arrow_end = self.b_arrow_length
        bx, by = self.ref_to_pixels(arrow_end, 0.0)
        self.canvas.create_line(
            ox,
            oy,
            bx,
            by,
            fill="#5d4037",
            width=2,
            dash=(8, 4),
            arrow=tk.LAST,
            tags="dynamic",
        )
        self.canvas.create_text(
            bx,
            by - 14,
            text=f"{b_name}  (+x)",
            font=("Arial", 10, "bold"),
            fill="#5d4037",
            tags="dynamic",
        )

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
            label = f"{path.name}.wp{i + 1}"
            self.canvas.create_text(
                px,
                py + 14,
                text=label,
                font=("Arial", 9, "bold"),
                fill=outline,
                tags="dynamic",
            )

    # ------------------------------------------------------------------
    # TF broadcast loop
    # ------------------------------------------------------------------
    def _broadcast_loop(self):
        rate = rospy.Rate(self.broadcast_rate_hz)
        while not rospy.is_shutdown() and not self._shutdown_event.is_set():
            try:
                self._broadcast_tick()
            except Exception as exc:  # noqa: BLE001
                rospy.logwarn_throttle(
                    5.0, f"[WaypointGUI] Broadcast tick failed: {exc}"
                )
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

    def _broadcast_tick(self):
        now = rospy.Time.now()
        transforms = []

        with self._paths_lock:
            paths_snapshot = list(self.paths)

        sim_transforms = self._collect_simulated_transforms(paths_snapshot, now)
        transforms.extend(sim_transforms)

        per_path_status = {}
        for path in paths_snapshot:
            if not path.publish_enabled.get():
                per_path_status[path] = ("paused", len(path.waypoints))
                continue

            composite_tf = self._build_composite_transform(path, now)
            if composite_tf is None:
                per_path_status[path] = ("waiting", len(path.waypoints))
                continue
            transforms.append(composite_tf)

            ref_frame_name = path.composite_frame_name()
            for i, wp in enumerate(path.waypoints):
                transforms.append(
                    self._build_wp_transform(path, i, wp, ref_frame_name, now)
                )
            per_path_status[path] = ("broadcasting", len(path.waypoints))

        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

        for path, (status, wp_count) in per_path_status.items():
            self._schedule_status_update(path, status, wp_count)

        self._rosparam_sync_counter += 1
        if self._rosparam_sync_counter >= 10:
            self._rosparam_sync_counter = 0
            self._sync_paths_rosparam(paths_snapshot)

    def _collect_simulated_transforms(self, paths_snapshot, now):
        if not self.simulate_var.get():
            return []

        transforms = []
        seen = set()
        slot = 0
        for path in paths_snapshot:
            for frame_name in (path.ref_a.get(), path.ref_b.get()):
                if not frame_name or frame_name in seen:
                    continue
                seen.add(frame_name)
                try:
                    self.tf_buffer.lookup_transform(
                        self.world_frame,
                        frame_name,
                        rospy.Time(0),
                        rospy.Duration(0.05),
                    )
                    continue
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    pass
                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = self.world_frame
                t.child_frame_id = frame_name
                t.transform.translation.x = 5.0 + 3.0 * slot
                t.transform.translation.y = 0.0
                t.transform.translation.z = 0.0
                t.transform.rotation.w = 1.0
                transforms.append(t)
                slot += 1
        return transforms

    def _schedule_status_update(self, path, status, wp_count):
        def _update():
            if not hasattr(path, "_status_label"):
                return
            try:
                if status == "broadcasting":
                    text = (
                        f"● Broadcasting  {path.composite_frame_name()} "
                        f"(+ {wp_count} waypoint{'s' if wp_count != 1 else ''})"
                    )
                    color = "#2e7d32"
                elif status == "paused":
                    text = "⏸ Publish disabled (toggle checkbox in ref panel)"
                    color = "#757575"
                else:
                    a = path.ref_a.get() or "?"
                    b = path.ref_b.get() or "?"
                    if self.simulate_var.get():
                        text = (
                            f"⚠ A→B lookup failed for {a}→{b} "
                            f"(simulate mode on — frames may still be wiring up)"
                        )
                    else:
                        text = (
                            f"⚠ A or B not in TF ({a}→{b}). "
                            f"Enable 'Simulate missing A/B frames' above to test offline."
                        )
                    color = "#d84315"
                path._status_label.config(text=text, fg=color)
            except tk.TclError:
                pass

        try:
            self.root.after(0, _update)
        except RuntimeError:
            pass

    def _sync_paths_rosparam(self, paths_snapshot):
        data = {}
        for path in paths_snapshot:
            data[path.name] = {
                "ref_a": path.ref_a.get(),
                "ref_b": path.ref_b.get(),
                "reference_frame": path.composite_frame_name(),
                "waypoints": len(path.waypoints),
                "waypoint_frames": [
                    path.wp_frame_name(i) for i in range(len(path.waypoints))
                ],
                "publish_enabled": bool(path.publish_enabled.get()),
            }
        try:
            rospy.set_param(self.paths_rosparam, data)
        except Exception as exc:  # noqa: BLE001
            rospy.logwarn_throttle(
                5.0, f"[WaypointGUI] Failed to publish {self.paths_rosparam}: {exc}"
            )

    def _build_composite_transform(self, path, now):
        """Broadcast <path>_ref as child of A frame.

        Transform from A → <path>_ref:
            translation = 0 (coincident with A)
            rotation    = yaw such that +x of <path>_ref points toward B

        A and B are both expressed in A's frame by looking them up; B's position
        in A's frame directly gives the +x direction. If either lookup fails the
        composite isn't broadcast this tick.
        """
        a = path.ref_a.get()
        b = path.ref_b.get()
        if not a or not b or a == b:
            return None

        try:
            tf_a_b = self.tf_buffer.lookup_transform(
                a,
                b,
                rospy.Time(0),
                rospy.Duration(0.1),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return None

        bx = tf_a_b.transform.translation.x
        by = tf_a_b.transform.translation.y
        if abs(bx) < 1e-9 and abs(by) < 1e-9:
            return None
        yaw = math.atan2(by, bx)

        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = a
        t.child_frame_id = path.composite_frame_name()
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        return t

    def _build_wp_transform(self, path, i, wp, ref_frame_name, now):
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = ref_frame_name
        t.child_frame_id = path.wp_frame_name(i)
        t.transform.translation.x = float(wp["x"])
        t.transform.translation.y = float(wp["y"])
        t.transform.translation.z = float(wp["z"])
        q = quaternion_from_euler(0.0, 0.0, math.radians(float(wp["yaw"])))
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        return t

    def _on_close(self):
        self._shutdown_event.set()
        try:
            if rospy.has_param(self.paths_rosparam):
                rospy.delete_param(self.paths_rosparam)
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()


def main():
    rospy.init_node("waypoint_gui", anonymous=True)
    root = tk.Tk()
    WaypointGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

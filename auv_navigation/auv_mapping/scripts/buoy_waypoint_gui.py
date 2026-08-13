#!/usr/bin/env python3

import math
import signal
import tkinter as tk
from tkinter import messagebox, ttk

import rospy
from auv_msgs.srv import SetBuoyWaypoints, SetBuoyWaypointsRequest


class BuoyWaypointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Buoy ve Surface Waypoint Arayüzü")
        self.root.geometry("1180x720")
        self.root.minsize(980, 640)
        self._is_closing = False
        self._connection_job = None
        self._map_points = []

        self.service_name = rospy.get_param(
            "~set_waypoints_service", "map/set_buoy_waypoints"
        )
        self.use_geodetic_var = tk.BooleanVar(value=True)
        self.coordinate_vars = {
            "start": {
                "latitude": tk.StringVar(),
                "longitude": tk.StringVar(),
                "x": tk.StringVar(value="0.000"),
                "y": tk.StringVar(value="0.000"),
            },
            "buoy": {
                "latitude": tk.StringVar(),
                "longitude": tk.StringVar(),
                "x": tk.StringVar(),
                "y": tk.StringVar(),
            },
            "surface": {
                "latitude": tk.StringVar(),
                "longitude": tk.StringVar(),
                "x": tk.StringVar(),
                "y": tk.StringVar(),
            },
        }

        self._configure_styles()
        self._build_layout()
        self._update_input_mode()
        self.root.after(200, self._check_service_connection)

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 20, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 11, "bold"))
        style.configure("Action.TButton", font=("TkDefaultFont", 11, "bold"), padding=8)
        style.configure("Treeview", rowheight=27)

    def _build_layout(self):
        main = ttk.Frame(self.root, padding=14)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=0, minsize=510)
        main.columnconfigure(1, weight=1)

        ttk.Label(
            main,
            text="Buoy / Surface Waypoint Konfigürasyonu",
            style="Title.TLabel",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        left = ttk.Frame(main)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)

        mode_frame = ttk.LabelFrame(
            left,
            text="Girdi Modu",
            style="Section.TLabelframe",
            padding=8,
        )
        mode_frame.grid(row=0, column=0, sticky="ew")
        ttk.Radiobutton(
            mode_frame,
            text="Lat / Lon",
            variable=self.use_geodetic_var,
            value=True,
            command=self._update_input_mode,
        ).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Radiobutton(
            mode_frame,
            text="Metre (debug)",
            variable=self.use_geodetic_var,
            value=False,
            command=self._update_input_mode,
        ).pack(side=tk.LEFT)

        self.coordinate_frame = ttk.LabelFrame(
            left,
            text="Koordinatlar",
            style="Section.TLabelframe",
            padding=10,
        )
        self.coordinate_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        for column in range(5):
            self.coordinate_frame.columnconfigure(column, weight=1 if column else 0)

        headers = ("Frame", "Enlem (N)", "Boylam (E)", "x / Kuzey (m)", "y / Batı (m)")
        for column, text in enumerate(headers):
            ttk.Label(
                self.coordinate_frame,
                text=text,
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=0, column=column, sticky="w", padx=4, pady=(0, 6))

        self.start_geodetic_entries = []
        self.target_geodetic_entries = []
        self.start_metre_entries = []
        self.target_metre_entries = []
        for row, (key, label) in enumerate(
            (("start", "Başlangıç"), ("buoy", "Buoy"), ("surface", "Surface")),
            start=1,
        ):
            variables = self.coordinate_vars[key]
            ttk.Label(self.coordinate_frame, text=label).grid(
                row=row, column=0, sticky="w", padx=4, pady=5
            )
            latitude_entry = ttk.Entry(
                self.coordinate_frame, textvariable=variables["latitude"], width=15
            )
            longitude_entry = ttk.Entry(
                self.coordinate_frame, textvariable=variables["longitude"], width=15
            )
            x_entry = ttk.Entry(
                self.coordinate_frame, textvariable=variables["x"], width=13
            )
            y_entry = ttk.Entry(
                self.coordinate_frame, textvariable=variables["y"], width=13
            )
            latitude_entry.grid(row=row, column=1, sticky="ew", padx=4, pady=5)
            longitude_entry.grid(row=row, column=2, sticky="ew", padx=4, pady=5)
            x_entry.grid(row=row, column=3, sticky="ew", padx=4, pady=5)
            y_entry.grid(row=row, column=4, sticky="ew", padx=4, pady=5)
            if key == "start":
                self.start_geodetic_entries.extend((latitude_entry, longitude_entry))
                self.start_metre_entries.extend((x_entry, y_entry))
            else:
                self.target_geodetic_entries.extend((latitude_entry, longitude_entry))
                self.target_metre_entries.extend((x_entry, y_entry))

        ttk.Label(
            left,
            text=(
                "Başlangıç Lat/Lon'u odom (0,0) ankrajıdır. Lat/Lon modunda Buoy ve "
                "Surface buna göre hesaplanır. +x kuzey, +y batıdır. Metre debug "
                "değerleri başlangıca göre doğrudan kullanılır."
            ),
            wraplength=490,
        ).grid(row=2, column=0, sticky="w", pady=(8, 8))

        action_row = ttk.Frame(left)
        action_row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        action_row.columnconfigure(1, weight=1)
        ttk.Button(
            action_row,
            text="Send to Vehicle",
            command=self.send_to_vehicle,
            style="Action.TButton",
        ).grid(row=0, column=0, sticky="w")
        self.connection_label = ttk.Label(
            action_row,
            text="ROS servisi kontrol ediliyor…",
            font=("TkDefaultFont", 9, "bold"),
        )
        self.connection_label.grid(row=0, column=1, sticky="e", padx=(8, 0))

        result_frame = ttk.LabelFrame(
            left,
            text="Araç Tarafında Çözülen Relatif Konumlar",
            style="Section.TLabelframe",
            padding=8,
        )
        result_frame.grid(row=4, column=0, sticky="nsew")
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        columns = ("frame", "x", "y", "distance")
        self.result_tree = ttk.Treeview(
            result_frame, columns=columns, show="headings", height=5
        )
        self.result_tree.heading("frame", text="TF Frame")
        self.result_tree.heading("x", text="x / Kuzey (m)")
        self.result_tree.heading("y", text="y / Batı (m)")
        self.result_tree.heading("distance", text="Başlangıca Uzaklık (m)")
        self.result_tree.column("frame", width=120, anchor="center")
        self.result_tree.column("x", width=115, anchor="e")
        self.result_tree.column("y", width=115, anchor="e")
        self.result_tree.column("distance", width=145, anchor="e")
        self.result_tree.grid(row=0, column=0, sticky="nsew")

        map_frame = ttk.LabelFrame(
            main,
            text="Yerel Harita (+x Kuzey, +y Batı)",
            style="Section.TLabelframe",
            padding=8,
        )
        map_frame.grid(row=1, column=1, sticky="nsew")
        map_frame.rowconfigure(0, weight=1)
        map_frame.columnconfigure(0, weight=1)
        self.map_canvas = tk.Canvas(
            map_frame,
            background="#eef7fb",
            highlightthickness=0,
        )
        self.map_canvas.grid(row=0, column=0, sticky="nsew")
        self.map_canvas.bind("<Configure>", lambda _event: self._draw_map())

    def _update_input_mode(self):
        use_geodetic = self.use_geodetic_var.get()
        for entry in self.start_geodetic_entries:
            entry.configure(state="normal")
        for entry in self.target_geodetic_entries:
            entry.configure(state="normal" if use_geodetic else "disabled")
        for entry in self.start_metre_entries:
            entry.configure(state="readonly")
        for entry in self.target_metre_entries:
            entry.configure(state="readonly" if use_geodetic else "normal")

    @staticmethod
    def _parse_number(raw_value, label):
        normalized = str(raw_value).strip().replace(",", ".")
        if not normalized:
            raise ValueError(f"{label} boş bırakılamaz.")
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError(f"{label} geçerli bir sayı değil.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} sonlu bir sayı olmalı.")
        return value

    def _parse_coordinate(self, raw_value, label, minimum, maximum):
        value = self._parse_number(raw_value, label)
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} [{minimum}, {maximum}] aralığında olmalı.")
        return value

    def _build_request(self):
        request = SetBuoyWaypointsRequest()
        request.use_geodetic = bool(self.use_geodetic_var.get())
        request.start_latitude_deg = self._parse_coordinate(
            self.coordinate_vars["start"]["latitude"].get(),
            "Başlangıç enlemi",
            -90.0,
            90.0,
        )
        request.start_longitude_deg = self._parse_coordinate(
            self.coordinate_vars["start"]["longitude"].get(),
            "Başlangıç boylamı",
            -180.0,
            180.0,
        )
        if request.use_geodetic:
            request.buoy_latitude_deg = self._parse_coordinate(
                self.coordinate_vars["buoy"]["latitude"].get(),
                "Buoy enlemi",
                -90.0,
                90.0,
            )
            request.buoy_longitude_deg = self._parse_coordinate(
                self.coordinate_vars["buoy"]["longitude"].get(),
                "Buoy boylamı",
                -180.0,
                180.0,
            )
            request.surface_latitude_deg = self._parse_coordinate(
                self.coordinate_vars["surface"]["latitude"].get(),
                "Surface enlemi",
                -90.0,
                90.0,
            )
            request.surface_longitude_deg = self._parse_coordinate(
                self.coordinate_vars["surface"]["longitude"].get(),
                "Surface boylamı",
                -180.0,
                180.0,
            )
        else:
            request.buoy_x_m = self._parse_number(
                self.coordinate_vars["buoy"]["x"].get(), "Buoy x"
            )
            request.buoy_y_m = self._parse_number(
                self.coordinate_vars["buoy"]["y"].get(), "Buoy y"
            )
            request.surface_x_m = self._parse_number(
                self.coordinate_vars["surface"]["x"].get(), "Surface x"
            )
            request.surface_y_m = self._parse_number(
                self.coordinate_vars["surface"]["y"].get(), "Surface y"
            )
        return request

    def _check_service_connection(self):
        if self._is_closing or rospy.is_shutdown():
            return
        try:
            rospy.wait_for_service(self.service_name, timeout=0.05)
            self.connection_label.configure(
                text=f"Bağlı: {rospy.resolve_name(self.service_name)}",
                foreground="#087f23",
            )
        except rospy.ROSException:
            self.connection_label.configure(
                text="Bağlantı yok — buoy_waypoint_publisher bekleniyor",
                foreground="#b00020",
            )
        self._connection_job = self.root.after(1000, self._check_service_connection)

    def send_to_vehicle(self):
        try:
            request = self._build_request()
            rospy.wait_for_service(self.service_name, timeout=2.0)
            service = rospy.ServiceProxy(self.service_name, SetBuoyWaypoints)
            response = service(request)
            if not response.success:
                messagebox.showerror("Yükleme Başarısız", response.message)
                return

            points = list(
                zip(
                    response.frame_ids,
                    response.x_m,
                    response.y_m,
                    response.distances_from_start_m,
                )
            )
            self._show_results(points)
            if request.use_geodetic:
                self._fill_computed_metres(points)
            self._map_points = points
            self._draw_map()
            messagebox.showinfo(
                "Yükleme Tamamlandı",
                "Buoy ve surface frame'leri araca yüklendi.",
            )
        except ValueError as exc:
            messagebox.showerror("Geçersiz Girdi", str(exc))
        except (rospy.ROSException, rospy.ServiceException) as exc:
            messagebox.showerror("ROS Servis Hatası", str(exc))

    def _show_results(self, points):
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for frame_id, x_m, y_m, distance_m in points:
            self.result_tree.insert(
                "",
                "end",
                values=(frame_id, f"{x_m:.3f}", f"{y_m:.3f}", f"{distance_m:.3f}"),
            )

    def _fill_computed_metres(self, points):
        points_by_frame = {frame_id: (x_m, y_m) for frame_id, x_m, y_m, _ in points}
        for key in ("buoy", "surface"):
            if key not in points_by_frame:
                continue
            x_m, y_m = points_by_frame[key]
            self.coordinate_vars[key]["x"].set(f"{x_m:.3f}")
            self.coordinate_vars[key]["y"].set(f"{y_m:.3f}")

    def _draw_map(self):
        canvas = self.map_canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 300)
        margin = 55

        canvas.create_line(42, 80, 42, 30, arrow=tk.LAST, width=3, fill="#263238")
        canvas.create_text(42, 18, text="N", font=("TkDefaultFont", 13, "bold"))
        canvas.create_text(100, 25, text="● Buoy", fill="#d50000", anchor="w")
        canvas.create_rectangle(92, 42, 104, 54, outline="#1565c0", width=2)
        canvas.create_text(110, 48, text="Surface", fill="#1565c0", anchor="w")

        if not self._map_points:
            canvas.create_text(
                width / 2,
                height / 2,
                text="Send to Vehicle sonrasında konumlar burada gösterilir",
                fill="#607d8b",
            )
            return

        map_points = [("Başlangıç", 0.0, 0.0, 0.0)] + self._map_points
        xs = [point[1] for point in map_points]
        ys = [point[2] for point in map_points]
        center_x_m = (min(xs) + max(xs)) / 2.0
        center_y_m = (min(ys) + max(ys)) / 2.0
        span_x_m = max(max(xs) - min(xs), 2.0)
        span_y_m = max(max(ys) - min(ys), 2.0)
        scale = (
            min(
                (height - 2 * margin) / span_x_m,
                (width - 2 * margin) / span_y_m,
            )
            * 0.75
        )

        def to_canvas(x_m, y_m):
            # North (+x) is screen-up; west (+y) is screen-left.
            return (
                width / 2.0 - (y_m - center_y_m) * scale,
                height / 2.0 - (x_m - center_x_m) * scale,
            )

        start_screen = to_canvas(0.0, 0.0)
        for point in self._map_points:
            target_screen = to_canvas(point[1], point[2])
            canvas.create_line(
                *start_screen,
                *target_screen,
                dash=(5, 4),
                fill="#78909c",
                width=2,
            )

        for frame_id, x_m, y_m, _distance_m in map_points:
            screen_x, screen_y = to_canvas(x_m, y_m)
            if frame_id == "Başlangıç":
                canvas.create_line(
                    screen_x - 7,
                    screen_y,
                    screen_x + 7,
                    screen_y,
                    width=3,
                    fill="#263238",
                )
                canvas.create_line(
                    screen_x,
                    screen_y - 7,
                    screen_x,
                    screen_y + 7,
                    width=3,
                    fill="#263238",
                )
            elif frame_id == "buoy":
                radius = 8
                canvas.create_oval(
                    screen_x - radius,
                    screen_y - radius,
                    screen_x + radius,
                    screen_y + radius,
                    fill="#d50000",
                    outline="#8e0000",
                    width=2,
                )
            else:
                half_size = 9
                canvas.create_rectangle(
                    screen_x - half_size,
                    screen_y - half_size,
                    screen_x + half_size,
                    screen_y + half_size,
                    fill="#90caf9",
                    outline="#1565c0",
                    width=3,
                )
            canvas.create_text(
                screen_x + 12,
                screen_y - 12,
                text=f"{frame_id}\n({x_m:.2f}, {y_m:.2f}) m",
                anchor="sw",
                font=("TkDefaultFont", 9, "bold"),
            )

    def close(self):
        if self._is_closing:
            return
        self._is_closing = True
        if self._connection_job is not None:
            self.root.after_cancel(self._connection_job)
        if not rospy.is_shutdown():
            rospy.signal_shutdown("Buoy waypoint GUI closed")
        self.root.quit()
        self.root.destroy()


def main():
    rospy.init_node("buoy_waypoint_gui", anonymous=True, disable_signals=True)
    root = tk.Tk()
    gui = BuoyWaypointGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.close)
    signal.signal(signal.SIGINT, lambda _signal, _frame: root.after(0, gui.close))
    root.mainloop()


if __name__ == "__main__":
    main()

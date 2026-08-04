#!/usr/bin/env python3

import signal
import tkinter as tk
from tkinter import messagebox, ttk

import rospy
from auv_msgs.srv import SetSurfaceMission, SetSurfaceMissionRequest


class SurfaceWaypointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Yüzey Waypoint Görevi")
        self.root.geometry("1040x670")
        self.root.minsize(920, 620)

        self.set_service_name = "map/set_surface_mission"
        self.service_connected = False
        self._connection_job = None

        self.start_longitude_var = tk.StringVar(value="29,25646091")
        self.start_latitude_var = tk.StringVar(value="40,86129602")
        self.visit_nearest_var = tk.BooleanVar(value=False)
        self.waypoint_vars = [
            {
                "longitude": tk.StringVar(value="29,25668898"),
                "latitude": tk.StringVar(value="40,86109848"),
                "camera": tk.BooleanVar(value=True),
            },
            {
                "longitude": tk.StringVar(),
                "latitude": tk.StringVar(),
                "camera": tk.BooleanVar(value=True),
            },
            {
                "longitude": tk.StringVar(),
                "latitude": tk.StringVar(),
                "camera": tk.BooleanVar(value=True),
            },
        ]

        self._configure_styles()
        self._build_layout()
        self._check_service_connection()

    def _configure_styles(self):
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 22, "bold"))
        style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 12, "bold"))
        style.configure("Status.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("Action.TButton", font=("TkDefaultFont", 11, "bold"), padding=8)
        style.configure("Treeview", rowheight=27)

    def _build_layout(self):
        container = ttk.Frame(self.root, padding=18)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        ttk.Label(
            container,
            text="Kullanıcı Arayüzü — Yüzey Waypoint Görevi",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w", pady=(0, 14))

        coordinate_frame = ttk.LabelFrame(
            container,
            text="Koordinatlar ve Kamera Konfigürasyonu",
            style="Section.TLabelframe",
            padding=12,
        )
        coordinate_frame.grid(row=1, column=0, sticky="ew")
        coordinate_frame.columnconfigure(1, weight=1)
        coordinate_frame.columnconfigure(2, weight=1)

        headers = ("Nokta", "Boylam (E)", "Enlem (N)", "Kamera")
        for column, text in enumerate(headers):
            ttk.Label(
                coordinate_frame, text=text, font=("TkDefaultFont", 10, "bold")
            ).grid(
                row=0,
                column=column,
                sticky="w",
                padx=6,
                pady=(0, 6),
            )

        ttk.Label(coordinate_frame, text="Başlangıç").grid(
            row=1, column=0, sticky="w", padx=6, pady=5
        )
        ttk.Entry(
            coordinate_frame, textvariable=self.start_longitude_var, width=24
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=5)
        ttk.Entry(
            coordinate_frame, textvariable=self.start_latitude_var, width=24
        ).grid(row=1, column=2, sticky="ew", padx=6, pady=5)
        ttk.Label(coordinate_frame, text="—").grid(
            row=1, column=3, sticky="w", padx=6, pady=5
        )

        for index, variables in enumerate(self.waypoint_vars, start=1):
            row = index + 1
            ttk.Label(coordinate_frame, text=f"Waypoint {index}").grid(
                row=row, column=0, sticky="w", padx=6, pady=5
            )
            ttk.Entry(
                coordinate_frame,
                textvariable=variables["longitude"],
                width=24,
            ).grid(row=row, column=1, sticky="ew", padx=6, pady=5)
            ttk.Entry(
                coordinate_frame,
                textvariable=variables["latitude"],
                width=24,
            ).grid(row=row, column=2, sticky="ew", padx=6, pady=5)
            self._build_camera_selector(
                coordinate_frame,
                row,
                variables["camera"],
            )

        self._build_visit_order_selector(container)

        info_text = (
            "Kamera Açık ise araç yüzeye çıktığında üç fotoğraf kaydedilir. "
            "Frame yönleri seçilen rota yönüne göre ayarlanır."
        )
        ttk.Label(container, text=info_text, wraplength=980).grid(
            row=3, column=0, sticky="w", pady=(10, 8)
        )

        button_frame = ttk.Frame(container)
        button_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(
            button_frame,
            text="Araca Yükle",
            command=self.send_to_vehicle,
            style="Action.TButton",
        ).grid(row=0, column=0, padx=(0, 8))

        self.connection_label = ttk.Label(
            button_frame,
            text="ROS servisi kontrol ediliyor…",
            style="Status.TLabel",
        )
        self.connection_label.grid(row=0, column=2, sticky="e")

        result_frame = ttk.LabelFrame(
            container,
            text="Başlangıca Göre Odom Sonuçları",
            style="Section.TLabelframe",
            padding=8,
        )
        result_frame.grid(row=5, column=0, sticky="nsew")
        container.rowconfigure(5, weight=1)
        result_frame.rowconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)

        columns = ("frame", "x", "y", "distance", "camera")
        self.result_tree = ttk.Treeview(
            result_frame,
            columns=columns,
            show="headings",
            height=4,
        )
        self.result_tree.heading("frame", text="TF Frame")
        self.result_tree.heading("x", text="x / Kuzey (m)")
        self.result_tree.heading("y", text="y / Batı (m)")
        self.result_tree.heading("distance", text="Başlangıca Uzaklık (m)")
        self.result_tree.heading("camera", text="Fotoğraf")
        self.result_tree.column("frame", width=210, anchor="center")
        self.result_tree.column("x", width=150, anchor="e")
        self.result_tree.column("y", width=150, anchor="e")
        self.result_tree.column("distance", width=210, anchor="e")
        self.result_tree.column("camera", width=110, anchor="center")
        self.result_tree.grid(row=0, column=0, sticky="nsew")

    @staticmethod
    def _build_camera_selector(parent, row, variable):
        selector = ttk.Frame(parent)
        selector.grid(row=row, column=3, sticky="w", padx=6, pady=5)

        selected_text = tk.StringVar()
        camera_on = tk.Button(
            selector,
            text="Açık",
            width=7,
            command=lambda: set_camera(True),
        )
        camera_on.grid(row=0, column=0, padx=(0, 4))
        camera_off = tk.Button(
            selector,
            text="Kapalı",
            width=7,
            command=lambda: set_camera(False),
        )
        camera_off.grid(row=0, column=1, padx=(0, 8))
        tk.Label(
            selector,
            textvariable=selected_text,
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=0, column=2, sticky="w")

        def set_camera(enabled):
            variable.set(enabled)
            update_selection()

        def update_selection():
            if variable.get():
                selected_text.set("Seçili: AÇIK")
                camera_on.configure(
                    background="#00c853",
                    activebackground="#00e676",
                    foreground="white",
                    relief=tk.SUNKEN,
                    borderwidth=3,
                )
                camera_off.configure(
                    background="#e6e6e6",
                    activebackground="#f0f0f0",
                    foreground="black",
                    relief=tk.RAISED,
                    borderwidth=1,
                )
            else:
                selected_text.set("Seçili: KAPALI")
                camera_on.configure(
                    background="#e6e6e6",
                    activebackground="#f0f0f0",
                    foreground="black",
                    relief=tk.RAISED,
                    borderwidth=1,
                )
                camera_off.configure(
                    background="#00c853",
                    activebackground="#00e676",
                    foreground="white",
                    relief=tk.SUNKEN,
                    borderwidth=3,
                )

        update_selection()

    def _build_visit_order_selector(self, parent):
        selector = ttk.LabelFrame(
            parent,
            text="Ziyaret Sırası",
            style="Section.TLabelframe",
            padding=8,
        )
        selector.grid(row=2, column=0, sticky="ew", pady=(10, 0))

        sequential_button = tk.Button(selector, text="1 → 2 → 3", width=18)
        nearest_button = tk.Button(selector, text="En yakın sonraki", width=18)
        status = tk.StringVar()

        def set_order(nearest_first):
            self.visit_nearest_var.set(nearest_first)
            if nearest_first:
                status.set("Seçili: Başlangıçtan ve her duraktan en yakın waypoint")
                nearest_button.configure(
                    background="#00c853",
                    activebackground="#00e676",
                    foreground="white",
                    relief=tk.SUNKEN,
                    borderwidth=3,
                )
                sequential_button.configure(
                    background="#e6e6e6",
                    foreground="black",
                    relief=tk.RAISED,
                    borderwidth=1,
                )
            else:
                status.set("Seçili: Waypoint 1 → Waypoint 2 → Waypoint 3")
                sequential_button.configure(
                    background="#00c853",
                    activebackground="#00e676",
                    foreground="white",
                    relief=tk.SUNKEN,
                    borderwidth=3,
                )
                nearest_button.configure(
                    background="#e6e6e6",
                    foreground="black",
                    relief=tk.RAISED,
                    borderwidth=1,
                )

        sequential_button.configure(command=lambda: set_order(False))
        nearest_button.configure(command=lambda: set_order(True))
        sequential_button.grid(row=0, column=0, padx=(0, 6))
        nearest_button.grid(row=0, column=1, padx=(0, 10))
        tk.Label(selector, textvariable=status, font=("TkDefaultFont", 9, "bold")).grid(
            row=0,
            column=2,
            sticky="w",
        )
        set_order(False)

    @staticmethod
    def _parse_coordinate(raw_value, label, minimum, maximum):
        normalized = str(raw_value).strip().replace(",", ".")
        if not normalized:
            raise ValueError(f"{label} boş bırakılamaz.")
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError(f"{label} geçerli bir sayı değil.") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{label} [{minimum}, {maximum}] aralığında olmalı.")
        return value

    def _build_request(self):
        request = SetSurfaceMissionRequest()
        request.start_longitude_deg = self._parse_coordinate(
            self.start_longitude_var.get(),
            "Başlangıç boylamı",
            -180.0,
            180.0,
        )
        request.start_latitude_deg = self._parse_coordinate(
            self.start_latitude_var.get(),
            "Başlangıç enlemi",
            -90.0,
            90.0,
        )

        request.waypoint_longitudes_deg = []
        request.waypoint_latitudes_deg = []
        request.camera_enabled = []
        request.visit_nearest_first = bool(self.visit_nearest_var.get())
        for index, variables in enumerate(self.waypoint_vars, start=1):
            request.waypoint_longitudes_deg.append(
                self._parse_coordinate(
                    variables["longitude"].get(),
                    f"Waypoint {index} boylamı",
                    -180.0,
                    180.0,
                )
            )
            request.waypoint_latitudes_deg.append(
                self._parse_coordinate(
                    variables["latitude"].get(),
                    f"Waypoint {index} enlemi",
                    -90.0,
                    90.0,
                )
            )
            request.camera_enabled.append(bool(variables["camera"].get()))
        return request

    def _check_service_connection(self):
        if rospy.is_shutdown():
            return
        try:
            rospy.wait_for_service(self.set_service_name, timeout=0.05)
            self.service_connected = True
            self.connection_label.configure(
                text=f"Bağlı: {rospy.resolve_name(self.set_service_name)}",
                foreground="#087f23",
            )
        except rospy.ROSException:
            self.service_connected = False
            self.connection_label.configure(
                text="Bağlantı yok — surface_waypoint_publisher bekleniyor",
                foreground="#b00020",
            )
        self._connection_job = self.root.after(1000, self._check_service_connection)

    def send_to_vehicle(self):
        try:
            request = self._build_request()
            rospy.wait_for_service(self.set_service_name, timeout=2.0)
            set_mission = rospy.ServiceProxy(
                self.set_service_name,
                SetSurfaceMission,
            )
            response = set_mission(request)
            if not response.success:
                messagebox.showerror("Yükleme Başarısız", response.message)
                return

            self._show_response_results(response, request.camera_enabled)
            messagebox.showinfo(
                "Yükleme Tamamlandı",
                "Üç waypoint ve kamera seçimleri araca yüklendi.\n"
                "TF koordinatları ve mesafeler aşağıdaki tabloda gösteriliyor.",
            )
        except ValueError as exc:
            messagebox.showerror("Geçersiz Koordinat", str(exc))
        except (rospy.ROSException, rospy.ServiceException) as exc:
            messagebox.showerror("ROS Servis Hatası", str(exc))

    def _show_response_results(self, response, camera_enabled):
        self._clear_results()
        camera_by_frame = {
            f"surface_waypoint_{index}": capture
            for index, capture in enumerate(camera_enabled, start=1)
        }
        for frame_id, x_m, y_m, distance_m in zip(
            response.waypoint_frame_ids,
            response.waypoint_x_m,
            response.waypoint_y_m,
            response.waypoint_distances_m,
        ):
            self.result_tree.insert(
                "",
                "end",
                values=(
                    frame_id,
                    f"{x_m:.3f}",
                    f"{y_m:.3f}",
                    f"{distance_m:.3f}",
                    "Açık" if camera_by_frame[frame_id] else "Kapalı",
                ),
            )

    def _clear_results(self):
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

    def close(self):
        if self._connection_job is not None:
            self.root.after_cancel(self._connection_job)
        rospy.signal_shutdown("Surface waypoint GUI closed")
        self.root.destroy()


def main():
    rospy.init_node("surface_waypoint_gui", anonymous=True, disable_signals=True)
    root = tk.Tk()
    gui = SurfaceWaypointGUI(root)
    root.protocol("WM_DELETE_WINDOW", gui.close)
    signal.signal(signal.SIGINT, lambda _signal, _frame: root.after(0, gui.close))
    root.mainloop()


if __name__ == "__main__":
    main()

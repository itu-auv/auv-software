#!/usr/bin/env python3
"""KDE Visualizer Module

Renders a top-down 2D density heatmap of KDE results for each object class
and publishes it as a compressed image. Runs on a dedicated daemon thread
so that rendering never blocks the KDE computation.
"""

import threading
import copy
import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage

CLASS_COLORS_BGR = [
    (80, 80, 255),  # soft red
    (80, 220, 80),  # green
    (255, 180, 50),  # blue-ish
    (180, 50, 255),  # magenta
    (220, 220, 50),  # cyan
    (50, 140, 255),  # orange
    (200, 100, 200),  # purple
    (100, 255, 255),  # yellow
    (255, 150, 150),  # light blue
    (150, 255, 150),  # light green
]


class KdeVisualizer:
    """Threaded KDE visualization publisher.

    Usage:
        viz = KdeVisualizer(image_width, image_height, class_colors, bandwidth, grid_resolution)
        viz.start()
        # ... after each KDE cycle:
        viz.update(kde_data, results)
    """

    def __init__(self, image_width, image_height, class_colors, bandwidth, grid_resolution):
        self.image_width = image_width
        self.image_height = image_height
        self.class_colors = class_colors
        self.bandwidth = bandwidth
        self.grid_resolution = grid_resolution

        self.image_pub = rospy.Publisher(
            "kde_map/visualization/compressed", CompressedImage, queue_size=1
        )

        # Threading primitives
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._kde_data = None
        self._results = None

        self._thread = threading.Thread(target=self._render_loop, daemon=True)

    def start(self):
        """Start the visualization render thread."""
        self._thread.start()

    def update(self, kde_data, results):
        """Submit new KDE data for rendering.

        This method returns immediately. The data is deep-copied under
        a lock and the render thread is signalled.
        """
        with self._lock:
            self._kde_data = copy.deepcopy(kde_data)
            self._results = copy.deepcopy(results)
        self._event.set()

    def _render_loop(self):
        """Main loop of the visualization thread."""
        while not rospy.is_shutdown():
            # Wait for new data (with timeout so we can check shutdown)
            signalled = self._event.wait(timeout=1.0)
            if not signalled:
                continue
            self._event.clear()

            # Grab latest data
            with self._lock:
                kde_data = self._kde_data
                results = self._results
                self._kde_data = None
                self._results = None

            if kde_data is None or not kde_data:
                continue

            if self.image_pub.get_num_connections() == 0:
                continue

            self._render_and_publish(kde_data, results)

    def _render_and_publish(self, kde_data, results):
        """Render the density heatmap and publish as CompressedImage."""
        img_w = self.image_width
        img_h = self.image_height
        padding = 60

        g_xmin = min(d["x_min"] for d in kde_data.values())
        g_xmax = max(d["x_max"] for d in kde_data.values())
        g_ymin = min(d["y_min"] for d in kde_data.values())
        g_ymax = max(d["y_max"] for d in kde_data.values())

        world_w = max(g_xmax - g_xmin, 0.1)
        world_h = max(g_ymax - g_ymin, 0.1)
        plot_w = img_w - 2 * padding
        plot_h = img_h - 2 * padding
        scale = min(plot_w / world_w, plot_h / world_h)

        # Centering offsets
        off_x = padding + (plot_w - world_w * scale) / 2.0
        off_y = padding + (plot_h - world_h * scale) / 2.0

        def w2p(wx, wy):
            """World coordinate → pixel coordinate."""
            u = int((wx - g_xmin) * scale + off_x)
            v = int(img_h - ((wy - g_ymin) * scale + off_y))
            return u, v

        canvas = np.full((img_h, img_w, 3), 30, dtype=np.uint8)

        # Draw grid lines
        grid_step = self._nice_grid_step(max(world_w, world_h))
        gx = np.arange(
            np.floor(g_xmin / grid_step) * grid_step,
            g_xmax + grid_step,
            grid_step,
        )
        gy = np.arange(
            np.floor(g_ymin / grid_step) * grid_step,
            g_ymax + grid_step,
            grid_step,
        )
        for x in gx:
            u, _ = w2p(x, 0)
            cv2.line(canvas, (u, padding), (u, img_h - padding), (50, 50, 50), 1)
            cv2.putText(
                canvas,
                f"{x:.1f}",
                (u - 10, img_h - padding + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (120, 120, 120),
                1,
            )
        for y in gy:
            _, v = w2p(0, y)
            cv2.line(canvas, (padding, v), (img_w - padding, v), (50, 50, 50), 1)
            cv2.putText(
                canvas,
                f"{y:.1f}",
                (5, v + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (120, 120, 120),
                1,
            )

        # ── Per-class density + points 
        legend_y = 25
        for cls_name, data in kde_data.items():
            color = self.class_colors.get(cls_name, (255, 255, 255))
            pts = data["points"]
            density = data["density"]

            # Draw density heatmap on a temporary layer
            if density.max() > 0:
                dn = (density / density.max() * 200).astype(np.uint8)
                # Map KDE grid corners to pixel space
                u_min, v_max_p = w2p(data["x_min"], data["y_min"])
                u_max, v_min_p = w2p(data["x_max"], data["y_max"])
                tw = max(abs(u_max - u_min), 1)
                th = max(abs(v_max_p - v_min_p), 1)

                dn_flipped = cv2.flip(dn, 0)
                dn_resized = cv2.resize(dn_flipped, (tw, th), interpolation=cv2.INTER_LINEAR)
                layer = np.zeros((th, tw, 3), dtype=np.uint8)
                for ch in range(3):
                    layer[:, :, ch] = (dn_resized * (color[ch] / 255.0)).astype(
                        np.uint8
                    )

                # Composite onto canvas with alpha blending
                y1 = max(v_min_p, 0)
                y2 = min(v_min_p + th, img_h)
                x1 = max(u_min, 0)
                x2 = min(u_min + tw, img_w)
                cy1, cy2 = y1 - v_min_p, y1 - v_min_p + (y2 - y1)
                cx1, cx2 = x1 - u_min, x1 - u_min + (x2 - x1)

                if y2 > y1 and x2 > x1 and cy2 > cy1 and cx2 > cx1:
                    roi = layer[cy1:cy2, cx1:cx2]
                    mask = np.any(roi > 15, axis=2)
                    croi = canvas[y1:y2, x1:x2]
                    blended = cv2.addWeighted(croi, 0.6, roi, 0.4, 0)
                    croi[mask] = blended[mask]

            # Draw raw detection points
            for pt in pts:
                u, v = w2p(pt[0], pt[1])
                if 0 <= u < img_w and 0 <= v < img_h:
                    cv2.circle(canvas, (u, v), 2, color, -1)

            # Draw peaks with labels
            if cls_name in results:
                short = cls_name.replace("_link", "")
                for i, (px, py, _pz, conf) in enumerate(results[cls_name]):
                    u, v = w2p(px, py)
                    cv2.circle(canvas, (u, v), 10, color, 2)
                    cv2.circle(canvas, (u, v), 3, (255, 255, 255), -1)
                    label = f"{short}#{i} d={conf:.3f}"
                    cv2.putText(
                        canvas,
                        label,
                        (u + 14, v - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        color,
                        1,
                    )

            # Legend entry
            short_name = cls_name.replace("_link", "")
            n_pts = len(pts)
            n_peaks = len(results.get(cls_name, []))
            cv2.rectangle(
                canvas,
                (img_w - 280, legend_y - 10),
                (img_w - 270, legend_y),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                f"{short_name} ({n_pts}pts, {n_peaks}pk)",
                (img_w - 265, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
            )
            legend_y += 18

        # ── Title & axes
        cv2.putText(
            canvas,
            "KDE Object Map",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"X:[{g_xmin:.1f},{g_xmax:.1f}]  Y:[{g_ymin:.1f},{g_ymax:.1f}]  "
            f"bw={self.bandwidth}m  res={self.grid_resolution}m",
            (10, img_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (150, 150, 150),
            1,
        )

        # ── Publish
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        success, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            msg.data = buf.tobytes()
            self.image_pub.publish(msg)

    @staticmethod
    def _nice_grid_step(span):
        """Choose a visually clean grid spacing for the given world span."""
        raw = span / 8.0
        mag = 10 ** int(np.floor(np.log10(max(raw, 1e-9))))
        ratio = raw / mag
        if ratio < 1.5:
            return mag
        elif ratio < 3.5:
            return 2 * mag
        elif ratio < 7.5:
            return 5 * mag
        else:
            return 10 * mag

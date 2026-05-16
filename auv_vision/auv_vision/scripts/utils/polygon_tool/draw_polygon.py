#!/usr/bin/env python3
"""
Polygon drawing tool for bottom camera forbidden masks.

Usage:
  python3 draw_polygon.py <image_path>

Controls:
  Left Click   - Add point to current polygon
  Right Click  - Finish current polygon & start new one
  'u'          - Undo last point
  'd'          - Delete last completed polygon
  'e'          - Export all polygons as Python code
  'q' / ESC    - Quit
"""

import sys
import cv2
import numpy as np

REFERENCE_WIDTH = 1920
REFERENCE_HEIGHT = 1080

COLORS = [
    (96, 69, 233),   # red-ish
    (136, 204, 68),  # green
    (255, 136, 68),  # blue
    (68, 136, 255),  # orange
    (255, 68, 204),  # magenta
    (0, 204, 204),   # cyan
]

polygons = []        # list of list of (x, y) tuples
current_points = []  # current polygon being drawn
img_original = None
window_name = "Polygon Tool (LClick=add, RClick=finish, E=export, Q=quit)"


def draw_all(img):
    overlay = img.copy()

    # Draw completed polygons
    for i, poly in enumerate(polygons):
        color = COLORS[i % len(COLORS)]
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (*color[:3],), lineType=cv2.LINE_AA)
        cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(poly):
            cv2.circle(img, (x, y), 4, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(j), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw current polygon in progress
    if current_points:
        color = COLORS[len(polygons) % len(COLORS)]
        pts = np.array(current_points, dtype=np.int32)
        if len(current_points) > 1:
            cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)
        for j, (x, y) in enumerate(current_points):
            cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(j), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Blend fill
    cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)

    # Status bar
    status = f"Polygons: {len(polygons)} | Current: {len(current_points)} pts"
    cv2.rectangle(img, (0, 0), (500, 28), (0, 0, 0), -1)
    cv2.putText(img, status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return img


def mouse_callback(event, x, y, flags, param):
    global current_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        refresh()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_points) >= 3:
            polygons.append(list(current_points))
            current_points = []
            print(f"[✓] Polygon {len(polygons)} finished with {len(polygons[-1])} points")
            refresh()
        else:
            print("[!] En az 3 nokta gerekli, polygon kapatılamadı.")


def refresh():
    display = draw_all(img_original.copy())
    cv2.imshow(window_name, display)


def export_python():
    if not polygons:
        print("\n[!] Henüz polygon yok!\n")
        return

    print("\n" + "=" * 60)
    print("# ---- COPY FROM HERE ----")
    print(f"BOTTOM_MASK_REFERENCE_WIDTH = {REFERENCE_WIDTH}.0")
    print(f"BOTTOM_MASK_REFERENCE_HEIGHT = {REFERENCE_HEIGHT}.0")
    print("BOTTOM_FORBIDDEN_MASKS = (")

    for i, poly in enumerate(polygons):
        print(f"    # Polygon {i + 1}")
        print("    (")
        for x, y in poly:
            print(f"        ({float(x)}, {float(y)}),")
        print("    ),")

    print(")")
    print("# ---- COPY END ----")
    print("=" * 60 + "\n")


def main():
    global img_original

    if len(sys.argv) < 2:
        print("Usage: python3 draw_polygon.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"[ERROR] Could not load image: {image_path}")
        sys.exit(1)

    h, w = img_original.shape[:2]
    print(f"[INFO] Image loaded: {w}x{h}")
    print(f"[INFO] Reference resolution: {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
    if w != REFERENCE_WIDTH or h != REFERENCE_HEIGHT:
        print(f"[WARN] Image size ({w}x{h}) differs from reference ({REFERENCE_WIDTH}x{REFERENCE_HEIGHT})!")
        print(f"       Coordinates will be in image pixel space.")
    print()
    print("Controls:")
    print("  Left Click  = Add point")
    print("  Right Click  = Finish polygon")
    print("  'u'          = Undo last point")
    print("  'd'          = Delete last polygon")
    print("  'e'          = Export Python code")
    print("  'q' / ESC    = Quit")
    print()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 1400), min(h, 800))
    cv2.setMouseCallback(window_name, mouse_callback)

    refresh()

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('u'):
            if current_points:
                removed = current_points.pop()
                print(f"[↩] Undo: removed point {removed}")
                refresh()
        elif key == ord('d'):
            if polygons:
                removed = polygons.pop()
                print(f"[🗑] Deleted polygon with {len(removed)} points")
                refresh()
        elif key == ord('e'):
            export_python()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Range, PointCloud2
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from std_msgs.msg import Header, Empty
from std_srvs.srv import SetBool, SetBoolResponse
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs import point_cloud2
import math
import numpy as np
from sklearn.cluster import DBSCAN
import threading


class PoolBoundaryDetector:
    def __init__(self):
        rospy.init_node("pool_boundary_detector", anonymous=True)

        self.max_point_cloud_size = rospy.get_param("~max_point_cloud_size", 25000)
        self.min_valid_range = rospy.get_param("~min_valid_range", 0.3)
        self.max_valid_range = rospy.get_param("~max_valid_range", 100.0)
        self.min_distance_between_points = rospy.get_param(
            "~min_distance_between_points", 0.02
        )
        self.odom_frame = rospy.get_param("~odom_frame", "odom")

        self.sonar_front_frame = rospy.get_param(
            "~sonar_front_frame", "taluy/base_link/sonar_front_link"
        )
        self.sonar_right_frame = rospy.get_param(
            "~sonar_right_frame", "taluy/base_link/sonar_right_link"
        )
        self.sonar_left_frame = rospy.get_param(
            "~sonar_left_frame", "taluy/base_link/sonar_left_link"
        )

        self.z_tolerance = rospy.get_param(
            "~z_tolerance", 0.05
        )  # Aynı z seviyesinde kabul etmek için tolerans
        self.cluster_distance = rospy.get_param(
            "~cluster_distance", 1
        )  # DBSCAN mesafe eşiği
        self.min_points_per_cluster = rospy.get_param(
            "~min_points_per_cluster", 5
        )  # Küme için min nokta sayısı
        self.line_fit_distance = rospy.get_param(
            "~line_fit_distance", 0.5
        )  # RANSAC ile çizgi uydurma mesafesi

        self.line_width = 0.1
        self.line_color_r = 1.0
        self.line_color_g = 0.0
        self.line_color_b = 0.0
        self.line_color_a = 1.0

        self.services_active = False
        self.lock = threading.RLock()

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.point_cloud_data = []
        self.wall_lines = []
        self.reference_z = None  # Referans z değeri (ilk ölçüm ile belirlenecek)
        self.point_count_since_last_detection = (
            0  # Son sınır tespitinden sonra eklenen nokta sayısı
        )

        self.pc_publisher = rospy.Publisher(
            "/sonar/point_cloud", PointCloud2, queue_size=10
        )
        self.boundary_publisher = rospy.Publisher(
            "/sonar/pool_boundaries", MarkerArray, queue_size=10
        )

        self.start_service = rospy.Service(
            "/sonar/start_mapping", SetBool, self.handle_start_service
        )
        self.clear_service = rospy.Service(
            "/sonar/clear_points", SetBool, self.handle_clear_service
        )

        self.sonar_front_subscriber = None
        self.sonar_right_subscriber = None
        self.sonar_left_subscriber = None

    def handle_start_service(self, req):
        with self.lock:
            if req.data:  # Başlat
                if not self.services_active:
                    self.services_active = True
                    # Ön sonar abone ol
                    self.sonar_front_subscriber = rospy.Subscriber(
                        "/taluy/sensors/sonar_front/data",
                        Range,
                        self.sonar_front_callback,
                    )
                    # Sağ sonar abone ol
                    self.sonar_right_subscriber = rospy.Subscriber(
                        "/taluy/sensors/sonar_right/data",
                        Range,
                        self.sonar_right_callback,
                    )
                    # Sol sonar abone ol
                    self.sonar_left_subscriber = rospy.Subscriber(
                        "/taluy/sensors/sonar_left/data",
                        Range,
                        self.sonar_left_callback,
                    )
                    rospy.loginfo(
                        "Pool boundary detection started with all three sonars"
                    )
                    return SetBoolResponse(success=True, message="Service started")
                else:
                    return SetBoolResponse(
                        success=False, message="Service is already running"
                    )
            else:  # Durdur
                if self.services_active:
                    self.services_active = False
                    # Tüm abonelikleri kapat
                    if self.sonar_front_subscriber:
                        self.sonar_front_subscriber.unregister()
                        self.sonar_front_subscriber = None
                    if self.sonar_right_subscriber:
                        self.sonar_right_subscriber.unregister()
                        self.sonar_right_subscriber = None
                    if self.sonar_left_subscriber:
                        self.sonar_left_subscriber.unregister()
                        self.sonar_left_subscriber = None
                    rospy.loginfo("Pool boundary detection stopped")
                    return SetBoolResponse(success=True, message="Service stopped")
                else:
                    return SetBoolResponse(
                        success=False, message="Service is already stopped"
                    )

    def handle_clear_service(self, req):
        with self.lock:
            if req.data:
                self.point_cloud_data = []
                self.wall_lines = []
                self.reference_z = None
                self.point_count_since_last_detection = 0
                rospy.loginfo("Point cloud data cleared")
                self.publish_point_cloud()
                self.publish_boundaries()
                return SetBoolResponse(success=True, message="Point cloud cleared")
            return SetBoolResponse(success=False, message="No action taken")

    def get_transform(self, target_frame, source_frame, timestamp):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, timestamp, timeout=rospy.Duration(0.1)
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            rospy.logwarn_throttle(10, f"Transform error: {str(e)}")
            return None

    def transform_point_to_odom(self, point_stamped):
        transform = self.get_transform(
            self.odom_frame, point_stamped.header.frame_id, point_stamped.header.stamp
        )

        if transform:
            return tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return None

    def is_point_too_close(self, new_point):
        new_point_array = [new_point.x, new_point.y, new_point.z]
        for existing_point in self.point_cloud_data:
            distance = math.sqrt(
                (new_point_array[0] - existing_point[0]) ** 2
                + (new_point_array[1] - existing_point[1]) ** 2
                + (new_point_array[2] - existing_point[2]) ** 2
            )
            if distance < self.min_distance_between_points:
                return True
        return False

    def process_sonar_reading(self, msg, sonar_frame):
        if not self.services_active:
            return

        if not (self.min_valid_range < msg.range < self.max_valid_range):
            return

        sensor_point = PointStamped()
        sensor_point.header.stamp = msg.header.stamp
        sensor_point.header.frame_id = sonar_frame
        sensor_point.point.x = msg.range
        sensor_point.point.y = 0.0
        sensor_point.point.z = 0.0

        odom_point = self.transform_point_to_odom(sensor_point)
        if not odom_point:
            return

        if self.is_point_too_close(odom_point.point):
            return

        if self.reference_z is None:
            self.reference_z = odom_point.point.z
            rospy.loginfo(f"Reference Z set to: {self.reference_z}")

        if len(self.point_cloud_data) >= self.max_point_cloud_size:
            self.point_cloud_data.pop(0)

        self.point_cloud_data.append(
            [odom_point.point.x, odom_point.point.y, odom_point.point.z]
        )

        self.point_count_since_last_detection += 1

        if self.point_count_since_last_detection >= 15:
            self.detect_pool_boundaries()
            self.point_count_since_last_detection = 0

            self.publish_point_cloud()
            self.publish_boundaries()

    def sonar_front_callback(self, msg):
        with self.lock:
            self.process_sonar_reading(msg, self.sonar_front_frame)

    def sonar_right_callback(self, msg):
        with self.lock:
            self.process_sonar_reading(msg, self.sonar_right_frame)

    def sonar_left_callback(self, msg):
        with self.lock:
            self.process_sonar_reading(msg, self.sonar_left_frame)

    def detect_pool_boundaries(self):
        if len(self.point_cloud_data) < self.min_points_per_cluster:
            return

        # Z değerine göre filtreleme yap
        filtered_points = []
        if self.reference_z is not None:
            for point in self.point_cloud_data:
                if abs(point[2] - self.reference_z) < self.z_tolerance:
                    filtered_points.append(point)
        else:
            filtered_points = self.point_cloud_data

        if len(filtered_points) < self.min_points_per_cluster:
            return

        # Numpy dizisine dönüştür
        points_array = np.array(filtered_points)

        # Sadece x-y düzleminde kümeleme yap
        points_xy = points_array[:, 0:2]

        # DBSCAN kümelemesi uygula
        clustering = DBSCAN(
            eps=self.cluster_distance, min_samples=self.min_points_per_cluster
        ).fit(points_xy)
        labels = clustering.labels_

        # Kümeleri işle
        unique_labels = set(labels)
        self.wall_lines = []

        for label in unique_labels:
            if label == -1:  # Gürültü
                continue

            # Küme noktalarını seç
            cluster_mask = labels == label
            cluster_points_xy = points_xy[cluster_mask]

            # RANSAC ile çizgi uydurma
            self.fit_line_to_points(cluster_points_xy)

    def fit_line_to_points(self, points):
        """RANSAC algoritması ile noktalara çizgi uydur"""
        if len(points) < 2:
            return

        # En iyi RANSAC modeli
        best_inliers = []
        best_line = None
        iterations = min(100, len(points) // 2)  # Maximum iterasyon sayısı

        for _ in range(iterations):
            # Rastgele 2 nokta seç
            indices = np.random.choice(len(points), 2, replace=False)
            p1, p2 = points[indices]

            # Bu iki noktadan geçen çizgi denklemi: ax + by + c = 0
            # y = mx + b şeklinde düzenlersek, m = -a/b, b = -c/b
            if p2[0] - p1[0] == 0:  # Dikey çizgi
                a, b, c = 1, 0, -p1[0]
            else:
                m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                a, b = -m, 1
                c = m * p1[0] - p1[1]

            # Normalize et
            norm = math.sqrt(a * a + b * b)
            a, b, c = a / norm, b / norm, c / norm

            # Çizgiye olan mesafeleri hesapla
            distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)

            # Inliers (çizgiye yakın noktalar)
            inliers = np.where(distances < self.line_fit_distance)[0]

            # Eğer bu daha iyi bir model ise, güncelle
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_line = (a, b, c)

        # Eğer yeterli inlier varsa
        if len(best_inliers) >= self.min_points_per_cluster // 2:
            # Son çizgi parametrelerini iyileştir
            inlier_points = points[best_inliers]

            # Lineer regresyon ile çizgiyi iyileştir
            x = inlier_points[:, 0]
            y = inlier_points[:, 1]

            # Dikey çizgi kontrolü
            x_range = np.max(x) - np.min(x)
            y_range = np.max(y) - np.min(y)

            if x_range < 0.1 * y_range:  # Dikey çizgi
                # x = sabit
                avg_x = np.mean(x)
                line_start = [avg_x, np.min(y)]
                line_end = [avg_x, np.max(y)]
            else:
                # y = mx + b
                m, b = np.polyfit(x, y, 1)

                # Çizgi sınırlarını belirle
                x_min, x_max = np.min(x), np.max(x)
                y_min = m * x_min + b
                y_max = m * x_max + b

                line_start = [x_min, y_min]
                line_end = [x_max, y_max]

            # Çizgiyi duvar olarak ekle
            self.wall_lines.append(line_start + line_end)  # [x1, y1, x2, y2]

    def publish_point_cloud(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.odom_frame
        pc2 = point_cloud2.create_cloud_xyz32(header, self.point_cloud_data)
        self.pc_publisher.publish(pc2)

    def publish_boundaries(self):
        if not self.wall_lines:
            return

        marker_array = MarkerArray()

        for i, line in enumerate(self.wall_lines):
            marker = Marker()
            marker.header.frame_id = self.odom_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "pool_boundaries"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.line_width  # Çizgi kalınlığı
            marker.color.r = self.line_color_r
            marker.color.g = self.line_color_g
            marker.color.b = self.line_color_b
            marker.color.a = self.line_color_a

            # Çizginin başlangıç ve bitiş noktaları
            p1 = Point()
            p1.x = line[0]
            p1.y = line[1]
            p1.z = self.reference_z if self.reference_z is not None else 0.0

            p2 = Point()
            p2.x = line[2]
            p2.y = line[3]
            p2.z = self.reference_z if self.reference_z is not None else 0.0

            marker.points = [p1, p2]
            marker_array.markers.append(marker)

        self.boundary_publisher.publish(marker_array)

    def run(self):
        rospy.loginfo(
            "Pool Boundary Detector node initialized with three sonars. Use services to start/stop."
        )
        rospy.spin()


if __name__ == "__main__":
    try:
        detector = PoolBoundaryDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node ended!")

#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Range, PointCloud2
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from std_msgs.msg import Header, Empty
from std_srvs.srv import SetBool, SetBoolResponse
from sensor_msgs import point_cloud2
import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.transform import Rotation
import threading

class SonarPointCloudGenerator:
    def __init__(self):
        # ROS düğümünü başlat
        rospy.init_node('sonar_pointcloud_generator', anonymous=True)
        
        # Parametreler
        self.max_point_cloud_size = rospy.get_param('~max_point_cloud_size', 5000)  # Maksimum nokta sayısı
        self.sonar_frame = rospy.get_param('~sonar_frame', 'taluy/base_link/sonar_front_link')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.min_valid_range = rospy.get_param('~min_valid_range', 0.3)
        self.max_valid_range = rospy.get_param('~max_valid_range', 100.0)
        self.min_distance_between_points = rospy.get_param('~min_distance_between_points', 0.02)  # 2 cm eşik değeri
        self.clustering_eps = rospy.get_param('~clustering_eps', 0.2)  # DBSCAN için mesafe eşiği
        self.clustering_min_samples = rospy.get_param('~clustering_min_samples', 5)  # DBSCAN için minimum örnek sayısı
        
        # Servis kilitleri
        self.services_active = False
        self.lock = threading.RLock()  # Reentrant kilit

        # TF2 ayarları
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Veri depolama
        self.point_cloud_data = []
        self.wall_clusters = []  # Duvar kümeleri

        # ROS arayüzleri
        self.pc_publisher = rospy.Publisher('/sonar/point_cloud', PointCloud2, queue_size=10)
        self.wall_publisher = rospy.Publisher('/sonar/wall_clusters', PointCloud2, queue_size=10)
        
        # Servisler
        self.start_service = rospy.Service('/sonar/start_mapping', SetBool, self.handle_start_service)
        self.clear_service = rospy.Service('/sonar/clear_points', SetBool, self.handle_clear_service)
        
        # Sonar subscriber'ı başlangıçta aktif değil
        self.sonar_subscriber = None

    def handle_start_service(self, req):
        """Servisi başlatma veya durdurma"""
        with self.lock:
            if req.data:  # Başlat
                if not self.services_active:
                    self.services_active = True
                    self.sonar_subscriber = rospy.Subscriber(
                        '/taluy/sensors/sonar_front/data', 
                        Range, 
                        self.sonar_callback
                    )
                    rospy.loginfo("Sonar point cloud generation started")
                    return SetBoolResponse(success=True, message="Service started")
                else:
                    return SetBoolResponse(success=False, message="Service is already running")
            else:  # Durdur
                if self.services_active:
                    self.services_active = False
                    if self.sonar_subscriber:
                        self.sonar_subscriber.unregister()
                        self.sonar_subscriber = None
                    rospy.loginfo("Sonar point cloud generation stopped")
                    return SetBoolResponse(success=True, message="Service stopped")
                else:
                    return SetBoolResponse(success=False, message="Service is already stopped")

    def handle_clear_service(self, req):
        """Tüm noktaları silme"""
        with self.lock:
            if req.data:
                self.point_cloud_data = []
                self.wall_clusters = []
                rospy.loginfo("Point cloud data cleared")
                # Boş bir point cloud yayınla
                self.publish_point_cloud()
                self.publish_wall_clusters()
                return SetBoolResponse(success=True, message="Point cloud cleared")
            return SetBoolResponse(success=False, message="No action taken")

    def get_transform(self, target_frame, source_frame, timestamp):
        """İki frame arasındaki transformasyonu güvenli şekilde al"""
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                timestamp,
                timeout=rospy.Duration(0.1)
            )
        except (tf2_ros.LookupException, 
                tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(10, f"Transform error: {str(e)}")
            return None

    def transform_point_to_odom(self, point_stamped):
        """Noktayı odom frame'ine dönüştür"""
        transform = self.get_transform(
            self.odom_frame,
            point_stamped.header.frame_id,
            point_stamped.header.stamp
        )
        
        if transform:
            return tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return None
    
    def is_point_too_close(self, new_point: Point) -> bool:
        """Yeni noktanın mevcut noktalara çok yakın olup olmadığını kontrol et"""
        for existing_point in self.point_cloud_data:
            distance = math.sqrt(
                (new_point.x - existing_point[0])**2 +
                (new_point.y - existing_point[1])**2 +
                (new_point.z - existing_point[2])**2
            )
            if distance < self.min_distance_between_points:
                return True  # Çok yakın, ekleme
        return False  # Yeterince uzak, ekleyebiliriz

    def detect_and_align_walls(self):
        """Duvarları tespit et ve hizala"""
        if len(self.point_cloud_data) < self.clustering_min_samples:
            return
            
        # Numpy dizisine dönüştür
        points_array = np.array(self.point_cloud_data)
        
        # DBSCAN kümelemesi uygula
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.clustering_min_samples).fit(points_array)
        labels = clustering.labels_
        
        # Her kümeyi işle
        unique_labels = set(labels)
        self.wall_clusters = []
        
        for label in unique_labels:
            if label == -1:  # Gürültü
                continue
                
            # Küme noktalarını seç
            cluster_points = points_array[labels == label]
            
            # Eğer yeterince nokta varsa, duvar olarak değerlendir
            if len(cluster_points) >= self.clustering_min_samples:
                # Temel Bileşen Analizi ile yönlendirme
                mean = np.mean(cluster_points, axis=0)
                centered_points = cluster_points - mean
                
                # Kovaryans matrisi ve özdeğerleri hesapla
                cov_matrix = np.cov(centered_points.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                
                # En büyük özdeğere sahip eigenvector, ana doğrultuyu verir
                main_axis = eigenvectors[:,np.argmax(eigenvalues)]
                
                # Hizalanmış noktaları duvar kümesine ekle
                for point in cluster_points:
                    self.wall_clusters.append([point[0], point[1], point[2]])

    def sonar_callback(self, msg):
        with self.lock:
            if not self.services_active:
                return
                
            # Ölçüm geçerlilik kontrolü
            if not (self.min_valid_range < msg.range < self.max_valid_range):
                return

            # Base_link frame'inde nokta oluştur
            sensor_point = PointStamped()
            sensor_point.header.stamp = msg.header.stamp
            sensor_point.header.frame_id = self.sonar_frame
            sensor_point.point.x = msg.range  # Sonar ölçümü base_link x ekseninde
            sensor_point.point.y = 0.0
            sensor_point.point.z = 0.0

            # Odom frame'ine dönüşüm
            odom_point = self.transform_point_to_odom(sensor_point)
            if not odom_point:
                return
            
            # Çok yakın nokta kontrolü
            if self.is_point_too_close(odom_point.point):
                return  # Çok yakın, ekleme

            # Maksimum nokta sayısı kontrolü
            if len(self.point_cloud_data) >= self.max_point_cloud_size:
                # En eski noktayı çıkar
                self.point_cloud_data.pop(0)

            # PointCloud verisine ekle
            self.point_cloud_data.append([
                odom_point.point.x,
                odom_point.point.y,
                odom_point.point.z
            ])

            # Periyodik olarak duvar tespiti yap (her 50 yeni noktada bir)
            if len(self.point_cloud_data) % 50 == 0:
                self.detect_and_align_walls()

            # PointCloud mesajlarını oluştur ve yayınla
            self.publish_point_cloud()
            if len(self.wall_clusters) > 0:
                self.publish_wall_clusters()

    def publish_point_cloud(self):
        """Tüm noktaları içeren point cloud'u yayınla"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.odom_frame
        pc2 = point_cloud2.create_cloud_xyz32(header, self.point_cloud_data)
        self.pc_publisher.publish(pc2)
        
    def publish_wall_clusters(self):
        """Duvar kümelerini içeren point cloud'u yayınla"""
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.odom_frame
        pc2 = point_cloud2.create_cloud_xyz32(header, self.wall_clusters)
        self.wall_publisher.publish(pc2)

    def run(self):
        rospy.loginfo("Sonar Point Cloud Generator node initialized. Use services to start/stop.")
        rospy.spin()

if __name__ == '__main__':
    try:
        generator = SonarPointCloudGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node sonlandırıldı!")
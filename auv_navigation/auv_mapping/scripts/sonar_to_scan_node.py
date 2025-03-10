#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import Range, PointCloud2
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import math

class SonarPointCloudGenerator:
    def __init__(self):
        # ROS düğümünü başlat
        rospy.init_node('sonar_pointcloud_generator', anonymous=True)
        
        # Parametreler
        self.cloud_history_size = float('inf')  # Artık noktaları silmeyeceğiz, ancak ihtiyaç olursa eski silme metodunu tekrar ekleyebiliriz.
        self.sonar_frame = 'taluy/base_link'
        self.odom_frame = 'odom'
        self.min_valid_range = 0.3
        self.max_valid_range = 100.0
        self.min_distance_between_points = 0.02  # 2 cm eşik değeri

        # TF2 ayarları
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Veri depolama
        self.point_cloud_data = []

        # ROS arayüzleri
        self.pc_publisher = rospy.Publisher('/sonar/point_cloud', PointCloud2, queue_size=10)
        rospy.Subscriber('/taluy/sensors/sonar_front/data', Range, self.sonar_callback)

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

    def sonar_callback(self, msg):
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
        
        #çok yakın nokta kontorlü
        if self.is_point_too_close(odom_point.point):
            return  # Çok yakın, ekleme

        # PointCloud verisine ekle
        self.point_cloud_data.append([
            odom_point.point.x,
            odom_point.point.y,
            odom_point.point.z
        ])

        # Tarihçe boyutunu yönet (gerekiyorsa, sonra ekleyebilirsiniz)
        # if len(self.point_cloud_data) > self.cloud_history_size:
        #     self.point_cloud_data = self.point_cloud_data[-self.cloud_history_size:]

        # PointCloud2 mesajı oluştur
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.odom_frame

        # PointCloud2'yi oluştur ve yayınla
        pc2 = point_cloud2.create_cloud_xyz32(header, self.point_cloud_data)
        self.pc_publisher.publish(pc2)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        generator = SonarPointCloudGenerator()
        generator.run()
    except rospy.ROSInterruptException:
        rospy.logerr("Node sonlandırıldı!")

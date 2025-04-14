#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import open3d as o3d
import struct
import tf2_ros
import tf2_sensor_msgs
from geometry_msgs.msg import TransformStamped

# message_filters kaldırıldı
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header


class DepthColorToPointCloudNode:
    def __init__(self):
        rospy.init_node('depth_color_to_pointcloud_node', anonymous=True)

        # --- Parametreler ---
        # Kamera iç parametreleri doğrudan atanır
        self.fx = 432.803545
        self.fy = 567.001237
        self.cx = 306.698087
        self.cy = 221.133655

        # Derinlik ölçek faktörü (metre cinsinden derinlik varsayılır)
        self.depth_scale = 1.0
        # Maksimum derinlik mesafesi (metre cinsinden)
        self.depth_trunc = 100.0
        # PointCloud için hedef Frame ID
        self.target_frame_id = "odom"
        # Noktaların başlangıçta referans aldığı frame ID (derinlik kamerasının optik frame'i)
        self.source_frame_id = "taluy/base_link" # Kullanıcı tarafından sağlandı

        rospy.loginfo(f"Kamera İç Parametreleri (Sabit): fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        rospy.loginfo(f"Derinlik Ölçeği: {self.depth_scale}, Derinlik Kesme: {self.depth_trunc}")
        rospy.loginfo(f"Kaynak Frame ID (Kamera Optik): {self.source_frame_id}, Hedef Frame ID: {self.target_frame_id}")

        # --- TF2 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- Subscriber'lar (Ayrı Callback'ler) ---
        self.latest_depth_msg = None
        self.latest_color_msg = None
        self.depth_sub = rospy.Subscriber(
            "/depth_any/camera/depth/image_raw",
            Image,
            self.depth_callback,
            queue_size=1 # En son mesajı işle
        )
        self.color_sub = rospy.Subscriber(
            '/taluy/cameras/cam_front/image_raw',
            Image,
            self.color_callback,
            queue_size=1 # En son mesajı işle
        )

        # --- Publisher ---
        self.pc_pub = rospy.Publisher('/point_cloud', PointCloud2, queue_size=2)

        # --- CV Bridge ---
        self.bridge = CvBridge()

        rospy.loginfo("Depth ve Color to PointCloud Node Başlatıldı (Ayrı Callback'ler)...")

    def color_callback(self, color_msg):
        """Sadece en son renkli görüntüyü saklar."""
        rospy.logdebug("Renkli görüntü callback çağrıldı.")
        self.latest_color_msg = color_msg

    def depth_callback(self, depth_msg):
        """Derinlik mesajı geldiğinde nokta bulutu oluşturmayı tetikler."""
        rospy.loginfo("--- depth_callback çağrıldı ---") # DEBUG
        self.latest_depth_msg = depth_msg

        # Henüz renkli görüntü gelmediyse işlem yapma
        if self.latest_color_msg is None:
            rospy.logwarn("Henüz renkli görüntü alınmadı, nokta bulutu oluşturulamıyor.")
            return

        # En son renkli görüntüyü kullan (zaman damgası farklı olabilir)
        color_msg = self.latest_color_msg

        # Zaman damgası farkını kontrol et (isteğe bağlı)
        time_diff = abs(depth_msg.header.stamp - color_msg.header.stamp).to_sec()
        rospy.logdebug(f"Derinlik ve Renkli görüntü zaman damgası farkı: {time_diff:.4f} s")
        try:
            rospy.loginfo(f"İşlenen Derinlik Mesajı Encoding: {depth_msg.encoding}") # DEBUG
            rospy.loginfo(f"Kullanılan Renkli Mesaj Encoding: {color_msg.encoding}") # DEBUG
            # ROS Image mesajlarını OpenCV/NumPy dizilerine dönüştür
            # Derinlik görüntüsü: Gelen encoding'e göre (16UC1 veya 32FC1)
            cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            # Renkli görüntü: Open3D için RGB'ye dönüştür
            cv_color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")

            rospy.loginfo(f"Görüntüler dönüştürüldü. Derinlik şekli: {cv_depth_image.shape if cv_depth_image is not None else 'None'}, tipi: {cv_depth_image.dtype if cv_depth_image is not None else 'None'}") # DEBUG
            rospy.loginfo(f"Renkli görüntü şekli: {cv_color_image.shape if cv_color_image is not None else 'None'}, tipi: {cv_color_image.dtype if cv_color_image is not None else 'None'}") # DEBUG

            # Derinlik görüntüsünü açıkça float32'ye dönüştür
            cv_depth_image = cv_depth_image.astype(np.float32)

            # Görüntülerin boş olmadığından emin olun
            if cv_depth_image is None or cv_color_image is None:
                rospy.logwarn("Boş görüntü(ler) alındı. Frame atlanıyor.")
                return

            # Derinlik görüntüsünün tek kanallı olduğundan emin olun
            if len(cv_depth_image.shape) != 2:
                 rospy.logwarn(f"Derinlik görüntüsü beklenmeyen şekle sahip {cv_depth_image.shape}. 2D dizi bekleniyordu. Frame atlanıyor.")
                 return

            # Renkli görüntünün 3 kanallı olduğundan emin olun
            if len(cv_color_image.shape) != 3 or cv_color_image.shape[2] != 3:
                 rospy.logwarn(f"Renkli görüntü beklenmeyen şekle sahip {cv_color_image.shape}. 3 kanal bekleniyordu. Frame atlanıyor.")
                 return

            # Boyutların eşleşip eşleşmediğini kontrol edin
            if cv_depth_image.shape[0] != cv_color_image.shape[0] or cv_depth_image.shape[1] != cv_color_image.shape[1]:
                rospy.logwarn(f"Derinlik ({cv_depth_image.shape}) ve renkli ({cv_color_image.shape}) görüntü boyutları eşleşmiyor. Frame atlanıyor.")
                # İsteğe bağlı olarak, uygunsa bir görüntüyü diğerine uyacak şekilde yeniden boyutlandırabilirsiniz
                # cv_color_image = cv2.resize(cv_color_image, (cv_depth_image.shape[1], cv_depth_image.shape[0]))
                return

            height, width = cv_depth_image.shape

            # NumPy dizilerini Open3D Image nesnelerine dönüştür
            # Open3D derinlik için float32 veya uint16 bekler. Gelen tipe göre kontrol et.
            if cv_depth_image.dtype == np.float32:
                # Eğer zaten float32 (metre) ise doğrudan kullan
                pass
            elif cv_depth_image.dtype == np.uint16:
                # Eğer uint16 (mm) ise float32'ye çevir (Open3D bunu doğrudan destekler)
                 pass # Open3D create_from_color_and_depth uint16'yı işleyebilir
            else:
                rospy.logwarn(f"Desteklenmeyen derinlik görüntüsü veri tipi: {cv_depth_image.dtype}. 16UC1 veya 32FC1 bekleniyor.")
                return

            o3d_depth = o3d.geometry.Image(cv_depth_image)
            o3d_color = o3d.geometry.Image(cv_color_image)


            # Renk ve derinlikten RGBDImage oluştur
            # convert_rgb_to_intensity=False renk bilgisini korur
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color,
                o3d_depth,
                depth_scale=self.depth_scale, # Eğer derinlik mm ise 1000.0, metre ise 1.0
                depth_trunc=self.depth_trunc, # Maksimum derinlik
                convert_rgb_to_intensity=False
            )

            # Open3D PinholeCamloeraIntrinsic nesnesi oluştur
            # Alınan görüntülerden görüntü boyutlarını kullanın
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, self.fx, self.fy, self.cx, self.cy)

            # RGBDImage ve iç parametrelerden PointCloud oluştur
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsic
                # Kamera pozisyonu biliniyorsa isteğe bağlı olarak dış parametre matrisi sağlayabilirsiniz
                # extrinsic = np.identity(4) # Örnek: Identity matrix
            )
            # Y ekseni etrafında 90 derece, ardından X ekseni etrafında 90 derece döndürme uygula
            pcd = pcd.transform(np.array([[1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, -1, 0],
                                            [0, 0, 0, 1]]))
            
            #o3d.visualization.draw_geometries([pcd])
            #pcd = pcd.voxel_down_sample(voxel_size=0.01)

            # İsteğe bağlı: İstatistiksel aykırı değerleri kaldır
            #pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

            # Open3D PointCloud'u ROS PointCloud2'ye dönüştür
            if not pcd.has_points():
                rospy.logwarn("Pointcloud does not have any points")
                return

            points_np = np.asarray(pcd.points)
            colors_np = (np.asarray(pcd.colors) * 255).astype(np.uint8) # 0-1 float'tan 0-255 uint8'e dönüştür

            # Nokta ve renk sayısının eşleşip eşleşmediğini kontrol edin
            if len(points_np) != len(colors_np):
                 rospy.logwarn(f"Nokta sayısı ({len(points_np)}) ve renk sayısı ({len(colors_np)}) eşleşmiyor. Yayınlama atlanıyor.")
                 return

            # PointCloud2 mesaj alanlarını hazırla
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1), # RGB'yi tek bir UINT32 alanına paketle
            ]

            # RGB verilerini tek bir UINT32 alanına paketle
            # colors_np'nin Nx3 olduğundan emin olun
            if colors_np.shape[1] != 3:
                 rospy.logwarn(f"Renk verisi beklenmeyen şekle sahip {colors_np.shape}. Nx3 bekleniyordu. Yayınlama atlanıyor.")
                 return

            packed_colors = np.zeros(len(colors_np), dtype=np.uint32)
            # Renkleri yinele ve paketle (bit kaydırma karmaşık dizilerden daha açık)
            for i in range(len(colors_np)):
                r, g, b = colors_np[i]
                # BGR'yi little-endian formatında uint32'ye paketle (ROS standardı)
                # RViz gibi araçlar genellikle BGR bekler
                rgb_packed = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0] # BGRX olarak paketle
                packed_colors[i] = rgb_packed
            
            points_float32 = points_np.astype(np.float32)
            packed_colors_float32 = packed_colors.astype(np.float32).reshape(-1, 1) # create_cloud float bekler
            points_list = []
            for i in range(len(points_np)):
                x, y, z = points_np[i]
                
                points_list.append([-points_float32[i, 2], points_float32[i, 0], points_float32[i, 1], packed_colors[i]]) # Doğru: uint32 kullan


            # PointCloud2 mesaj başlığı oluştur (Kaynak frame ile)
            header = Header()
            header.stamp = depth_msg.header.stamp
            # Başlangıçta noktaların referans aldığı frame'i kullan
            header.frame_id = self.source_frame_id

            # PointCloud2 mesajı oluştur (Kaynak frame'de)
            # points_list'i kullan
            rospy.loginfo(f"PointCloud2 mesajı oluşturuluyor ({self.source_frame_id}). Nokta sayısı: {len(points_list)}") # DEBUG
            cloud_msg_source = pc2.create_cloud(header, fields, points_list)

            # Transformasyonu uygula
            try:
                # Kaynak frame'den hedef frame'e dönüşümü al
                # En son mevcut dönüşümü almak için rospy.Time(0) kullan
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame_id,    # Hedef frame
                    self.source_frame_id,    # Kaynak frame
                    rospy.Time(0),           # En son mevcut zaman damgası
                    rospy.Duration(1.0)      # Bekleme süresi (isteğe bağlı, zaman aşımı için)
                )

                # PointCloud2 mesajını dönüştür
                rospy.loginfo(f"PointCloud2 mesajı {self.source_frame_id} -> {self.target_frame_id} frame'ine dönüştürülüyor...") # DEBUG
                cloud_msg_transformed = tf2_sensor_msgs.do_transform_cloud(cloud_msg_source, transform)

                # Dönüştürülmüş mesajı yayınla
                rospy.loginfo(f"Dönüştürülmüş PointCloud2 mesajı ({self.target_frame_id}) yayınlanıyor...") # DEBUG
                self.pc_pub.publish(cloud_msg_source)
                rospy.loginfo("Dönüştürülmüş PointCloud yayınlandı.") # DEBUG

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"TF dönüşümü alınamadı ({self.source_frame_id} -> {self.target_frame_id}): {e}")

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Hatası: {e}")
        except Exception as e:
            rospy.logerr(f"Görüntüleri işleme hatası: {e}")
            import traceback
            traceback.print_exc() # Hata ayıklama için ayrıntılı traceback yazdır

if __name__ == "__main__":
    try:
        node = DepthColorToPointCloudNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict
import math  # math.isnan ve math.isinf kontrolü için


class SonarClusterDetector:
    def __init__(self):
        rospy.init_node("sonar_cluster_detector", anonymous=True)

        # Parametreler
        self.dbscan_eps = 0.2  # Bir grup içindeki noktalar arası maks. uzaklık (10 cm)
        # "nokta grupları ... noktalardan oluşuyor" (çoğul) -> en az 2 nokta
        self.dbscan_min_samples = 2

        self.min_dist_between_groups = (
            1.0  # Grupların merkezleri arası min. uzaklık (1 metre)
        )
        self.max_dist_between_groups = (
            2.0  # Grupların merkezleri arası maks. uzaklık (2 metre)
        )

        # Yayınlama için gereken toplam grup sayısı aralığı
        self.min_total_groups_to_publish = 3
        self.max_total_groups_to_publish = 9

        # Abone Olunan Topic
        self.subscriber = rospy.Subscriber(
            "/sonar/point_cloud",
            PointCloud2,
            self.point_cloud_callback,
            queue_size=1,  # Eski mesajlarla uğraşmamak için kuyruk boyutu 1
        )

        # Yayın Yapılacak Topic
        self.publisher = rospy.Publisher(
            "/sonar/processed_point_cloud", PointCloud2, queue_size=10
        )

        rospy.loginfo(
            f"Sonar Cluster Detector düğümü başlatıldı. Parametreler: \n"
            f"  DBSCAN eps: {self.dbscan_eps}m, min_samples: {self.dbscan_min_samples}\n"
            f"  Gruplar arası uzaklık: [{self.min_dist_between_groups}m, {self.max_dist_between_groups}m]\n"
            f"  Yayın için gereken grup sayısı: [{self.min_total_groups_to_publish}, {self.max_total_groups_to_publish}]"
        )

    def point_cloud_callback(self, ros_cloud):
        rospy.logdebug(
            f"Nokta bulutu alındı: {ros_cloud.width * ros_cloud.height} nokta."
        )
        header = ros_cloud.header  # Orijinal mesajın header'ını sakla

        try:
            # PointCloud2'den noktaları oku (x, y, z, ... alanlarını içeren demetler halinde)
            points_data_iterator = pc2.read_points(ros_cloud, skip_nans=True)

            original_points_with_all_fields = []  # Orijinal tam nokta verileri
            xy_points_for_clustering = []  # Sadece X, Y koordinatları (kümeleme için)
            point_indices = []  # Orijinal nokta indekslerini takip et

            for idx, p in enumerate(points_data_iterator):
                # Standart olarak x, y, z ilk üç alandır. En az x ve y olmalı.
                if len(p) >= 2:
                    # NaN veya Inf değerleri ayrıca kontrol et (skip_nans genel bir önlem)
                    if not (
                        math.isnan(p[0])
                        or math.isinf(p[0])
                        or math.isnan(p[1])
                        or math.isinf(p[1])
                    ):
                        xy_points_for_clustering.append([p[0], p[1]])
                        original_points_with_all_fields.append(p)
                        point_indices.append(idx)
                # else: Noktanın yeterli alanı yoksa atlanacak
        except Exception as e:
            rospy.logerr(f"PointCloud2'den noktalar okunurken hata: {e}")
            return

        if not xy_points_for_clustering:
            rospy.logdebug("Kümeleme için geçerli XY noktası bulunamadı.")
            return

        xy_points_np = np.array(xy_points_for_clustering)
        rospy.logdebug(
            f"Kümeleme için {len(xy_points_for_clustering)} nokta hazırlandı."
        )

        # Aşama 1: DBSCAN ile ilk kümeleme (grup içi maks. 10cm X-Y uzaklığı)
        try:
            db = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                metric="euclidean",
            ).fit(xy_points_np)
        except Exception as e:
            rospy.logerr(f"DBSCAN kümeleme hatası: {e}")
            return

        labels = db.labels_  # Her nokta için küme etiketi (-1 gürültü demektir)

        unique_labels = set(labels)
        # Gürültü olmayan kümelerin sayısı
        num_initial_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        rospy.logdebug(
            f"DBSCAN ile {num_initial_clusters} adet başlangıç kümesi bulundu (gürültü hariç)."
        )

        if num_initial_clusters == 0:
            rospy.logdebug("DBSCAN tarafından gürültü olmayan küme bulunamadı.")
            return

        # Aşama 2: Noktaları küme etiketlerine göre grupla ve merkezlerini hesapla
        clusters_data = defaultdict(lambda: {"points_full_data": [], "xy_coords": []})
        for i, label in enumerate(labels):
            if label != -1:  # Gürültü değilse
                clusters_data[label]["points_full_data"].append(
                    original_points_with_all_fields[i]
                )
                clusters_data[label]["xy_coords"].append(xy_points_np[i])

        if not clusters_data:
            rospy.logdebug("Gürültü olmayan küme verisi toplanamadı.")
            return

        cluster_centroids = {}  # {etiket_id: merkez_koordinat_xy}
        # Gürültü olmayan ve boş olmayan kümelerin etiketleri
        valid_cluster_labels = sorted(
            [
                label
                for label in clusters_data.keys()
                if clusters_data[label]["xy_coords"]
            ]
        )

        for label in valid_cluster_labels:
            centroid_xy = np.mean(np.array(clusters_data[label]["xy_coords"]), axis=0)
            cluster_centroids[label] = centroid_xy

        rospy.logdebug(f"{len(valid_cluster_labels)} kümenin merkezi hesaplandı.")

        # Aşama 3: Kümeleri merkezler arası uzaklığa göre filtrele (1-2 metre)
        # Bir küme, diğer kümelerden en az birine 1-2m uzaklıktaysa "kalifiye" olur.
        qualified_labels_meeting_spacing_criteria = set()

        if (
            len(valid_cluster_labels) >= 2
        ):  # En az iki küme olmalı ki aralarındaki mesafe kontrol edilebilsin
            for i in range(len(valid_cluster_labels)):
                label_a = valid_cluster_labels[i]
                centroid_a = cluster_centroids[label_a]
                is_group_a_qualified_by_spacing = False
                for j in range(len(valid_cluster_labels)):
                    if i == j:  # Kendisiyle karşılaştırma yapma
                        continue
                    label_b = valid_cluster_labels[j]
                    centroid_b = cluster_centroids[label_b]

                    # X-Y düzleminde Öklid mesafesi
                    dist = np.linalg.norm(centroid_a - centroid_b)

                    if (
                        self.min_dist_between_groups
                        <= dist
                        <= self.max_dist_between_groups
                    ):
                        is_group_a_qualified_by_spacing = True
                        break  # A grubu kalifiye oldu, diğer B'lere bakmaya gerek yok

                if is_group_a_qualified_by_spacing:
                    qualified_labels_meeting_spacing_criteria.add(label_a)
        else:
            rospy.logdebug("Gruplar arası mesafe kontrolü için yeterli küme (<2) yok.")

        num_groups_after_spacing_filter = len(qualified_labels_meeting_spacing_criteria)
        rospy.logdebug(
            f"Gruplar arası mesafe kriterini sağlayan {num_groups_after_spacing_filter} grup bulundu."
        )

        # Aşama 4: Kalifiye grup sayısının istenen aralıkta [3, 9] olup olmadığını kontrol et
        if (
            self.min_total_groups_to_publish
            <= num_groups_after_spacing_filter
            <= self.max_total_groups_to_publish
        ):
            rospy.loginfo(
                f"{num_groups_after_spacing_filter} grup için veri yayınlanıyor."
            )

            final_points_to_publish_tuples = []
            for label in qualified_labels_meeting_spacing_criteria:
                final_points_to_publish_tuples.extend(
                    clusters_data[label]["points_full_data"]
                )

            rospy.loginfo(
                f"Yayınlanacak nokta sayısı: {len(final_points_to_publish_tuples)}"
            )

            if final_points_to_publish_tuples:
                # Orijinal header ve alan tanımlarını kullanarak PointCloud2 mesajı oluştur
                if not ros_cloud.fields:  # Gelen mesajda alan tanımı yoksa hata ver
                    rospy.logerr(
                        "Giriş PointCloud2 mesajında alan tanımı yok. Çıkış bulutu oluşturulamıyor."
                    )
                    return

                try:
                    # create_cloud yerine create_cloud_xyz32 kullanarak daha basit bir yaklaşım dene
                    # Eğer sadece XYZ verileri varsa:
                    if len(ros_cloud.fields) <= 3:
                        xyz_points = []
                        for point in final_points_to_publish_tuples:
                            if len(point) >= 3:
                                xyz_points.append([point[0], point[1], point[2]])
                            else:
                                xyz_points.append(
                                    [point[0], point[1], 0.0]
                                )  # Z eksikse 0 koy

                        output_cloud_msg = pc2.create_cloud_xyz32(header, xyz_points)
                    else:
                        # Daha karmaşık alan yapısı için orijinal yöntemi kullan
                        output_cloud_msg = pc2.create_cloud(
                            header, ros_cloud.fields, final_points_to_publish_tuples
                        )

                    # Mesaj boyutunu kontrol et
                    if output_cloud_msg.width == 0 or output_cloud_msg.height == 0:
                        rospy.logwarn("Oluşturulan PointCloud2 mesajı boş görünüyor.")
                        # Manuel olarak boyutları ayarla
                        output_cloud_msg.width = len(final_points_to_publish_tuples)
                        output_cloud_msg.height = 1
                        output_cloud_msg.is_dense = False

                    # Timestamp'i orijinal mesajdan koru (simülasyon zamanı uyumluluğu için)
                    output_cloud_msg.header.stamp = header.stamp

                    self.publisher.publish(output_cloud_msg)
                    rospy.loginfo(
                        f"{num_groups_after_spacing_filter} gruptan {len(final_points_to_publish_tuples)} nokta içeren PointCloud2 yayınlandı."
                    )

                    # Debug: Yayınlanan mesajın özelliklerini logla
                    rospy.logdebug(
                        f"Yayınlanan mesaj özellikleri: width={output_cloud_msg.width}, height={output_cloud_msg.height}, "
                        f"point_step={output_cloud_msg.point_step}, row_step={output_cloud_msg.row_step}"
                    )

                except Exception as e:
                    rospy.logerr(f"Çıkış PointCloud2 mesajı oluşturulurken hata: {e}")
                    # Hata durumunda alternatif basit mesaj göndermeyi dene
                    try:
                        rospy.logwarn(
                            "Alternatif basit XYZ PointCloud2 mesajı deneniyor..."
                        )
                        simple_xyz = [
                            [p[0], p[1], p[2] if len(p) > 2 else 0.0]
                            for p in final_points_to_publish_tuples
                        ]
                        simple_msg = pc2.create_cloud_xyz32(header, simple_xyz)
                        simple_msg.header.stamp = header.stamp
                        self.publisher.publish(simple_msg)
                        rospy.loginfo("Alternatif basit mesaj yayınlandı.")
                    except Exception as e2:
                        rospy.logerr(f"Alternatif mesaj da başarısız: {e2}")
            else:
                rospy.logdebug(
                    "Yayınlama koşulları sağlandı ancak yayınlanacak nokta bulunamadı."
                )
        else:
            rospy.logdebug(
                f"Kalifiye grup sayısı ({num_groups_after_spacing_filter}) "
                f"istenilen aralıkta [{self.min_total_groups_to_publish}, {self.max_total_groups_to_publish}] değil. Yayınlanmıyor."
            )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        detector = SonarClusterDetector()
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Sonar Cluster Detector düğümü kesildi ve kapatıldı.")
    except Exception as e:
        rospy.logfatal(f"SonarClusterDetector'da beklenmedik bir hata oluştu: {e}")

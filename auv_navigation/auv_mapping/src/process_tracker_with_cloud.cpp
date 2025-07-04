/**
 * auv_mapping
 * ROS node for processing PointCloud2 data with YOLO detections
 * Inspired by ultralytics_ros tracker_with_cloud_node
 */

#include <geometry_msgs/TransformStamped.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <ultralytics_ros/YoloResult.h>  // vision_msgs/Detection2DArray yerine
#include <vision_msgs/Detection3DArray.h>
#include <visualization_msgs/MarkerArray.h>

// PCL Kütüphaneleri
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>  // getMinMax3D için gerekli
#include <pcl/common/pca.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

class ProcessTrackerWithCloud {
 private:
  // ROS üyeleri
  ros::NodeHandle nh_, pnh_;

  // Parametreler
  std::string camera_info_topic_, lidar_topic_, yolo_result_topic_;
  std::string yolo_3d_result_topic_;
  float cluster_tolerance_, voxel_leaf_size_;
  int min_cluster_size_, max_cluster_size_;
  float roi_expansion_factor_;

  // Yayıncılar
  ros::Publisher detection_cloud_pub_;
  ros::Publisher object_transform_pub_;

  // Abonelikler ve senkronizasyon
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub_;
  message_filters::Subscriber<ultralytics_ros::YoloResult> yolo_result_sub_;

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::PointCloud2,
      ultralytics_ros::YoloResult>
      SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  boost::shared_ptr<Sync> sync_;

  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  tf2_ros::Buffer tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string camera_optical_frame_;
  std::string base_link_frame_;

  // Kamera modeli
  image_geometry::PinholeCameraModel cam_model_;

  // Son işlem zamanı (marker ömrü için)
  ros::Time last_call_time_;

  // Atlanacak tespit ID'lerini saklayan set
  std::set<int> skip_detection_ids;

  // Tespit ID'sinden prop ismini almak için bir mapping ekleyelim
  std::map<int, std::string> id_to_prop_name = {
      {8, "red_buoy_link"},       {7, "path_link"},
      {9, "bin_whole_link"},      {12, "torpedo_map_link"},
      {13, "torpedo_hole_link"},  {1, "gate_left_link"},
      {2, "gate_right_link"},     {3, "gate_blue_arrow_link"},
      {4, "gate_red_arrow_link"}, {5, "gate_middle_part_link"},
      {14, "octagon_link"}};

 public:
  ProcessTrackerWithCloud()
      : pnh_("~"),
        tf_buffer_(ros::Duration(10.0)),
        tf_listener_(std::make_unique<tf2_ros::TransformListener>(tf_buffer_)) {
    // Parametreleri yükle
    pnh_.param<std::string>("camera_info_topic", camera_info_topic_,
                            "camera_info");
    pnh_.param<std::string>("lidar_topic", lidar_topic_, "points_raw");
    pnh_.param<std::string>("yolo_result_topic", yolo_result_topic_,
                            "yolo_result");
    pnh_.param<float>("cluster_tolerance", cluster_tolerance_, 0.3);
    pnh_.param<float>("voxel_leaf_size", voxel_leaf_size_, 0.1);
    pnh_.param<int>("min_cluster_size", min_cluster_size_, 100);
    pnh_.param<int>("max_cluster_size", max_cluster_size_, 10000);
    pnh_.param<float>("roi_expansion_factor", roi_expansion_factor_,
                      1.1);  // %10 genişletme

    // Atlanacak tespit ID'lerini parametre olarak al (varsayılan olarak 7 ve 13
    // ID'leri atlanır)
    std::vector<int> skip_ids;
    pnh_.getParam("skip_detection_ids", skip_ids);
    if (skip_ids.empty()) {
      // Varsayılan değerleri ayarla
      skip_detection_ids = {7, 13};  // path_link ve torpedo_hole_link
    } else {
      // Parametre ile gelen değerleri kullan
      skip_detection_ids.clear();
      skip_detection_ids.insert(skip_ids.begin(), skip_ids.end());
    }

    // Yayıncıları başlat
    detection_cloud_pub_ =
        nh_.advertise<sensor_msgs::PointCloud2>("detection_cloud", 1);
    object_transform_pub_ = nh_.advertise<geometry_msgs::TransformStamped>(
        "/taluy/map/object_transform_updates", 10);

    // Abonelikleri başlat
    camera_info_sub_.subscribe(nh_, camera_info_topic_, 10);
    lidar_sub_.subscribe(nh_, lidar_topic_, 10);
    yolo_result_sub_.subscribe(nh_, yolo_result_topic_, 10);

    // Senkronizasyonu yapılandır
    sync_ = boost::make_shared<Sync>(SyncPolicy(10), camera_info_sub_,
                                     lidar_sub_, yolo_result_sub_);
    sync_->registerCallback(
        boost::bind(&ProcessTrackerWithCloud::syncCallback, this, _1, _2, _3));

    // TF broadcaster oluştur
    tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());
    pnh_.param<std::string>("camera_optical_frame", camera_optical_frame_,
                            "taluy/camera_depth_optical_frame");
    base_link_frame_ = "taluy/base_link";
  }

  // Senkronize mesajlar için callback fonksiyonu
  void syncCallback(
      const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
      const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
      const ultralytics_ros::YoloResultConstPtr& yolo_result_msg) {
    // Kamera modelini güncelle
    cam_model_.fromCameraInfo(camera_info_msg);

    // Çağrı zamanını kaydet
    ros::Time current_call_time = ros::Time::now();
    ros::Duration callback_interval = current_call_time - last_call_time_;
    last_call_time_ = current_call_time;

    // YOLO tespiti yoksa işlem yapma
    if (yolo_result_msg->detections.detections.empty()) {
      return;
    }

    // Nokta bulutunu PCL formatına dönüştür
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Nokta bulutu boşsa işlem yapma
    if (cloud->points.empty()) {
      return;
    }

    // Nokta bulutunu downsample et
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud =
        downsampleCloud(cloud);

    // İşlenmemiş nokta bulutu ve YOLO sonuçları için 3D tespitleri oluşturacak
    // veri yapılarını hazırla
    vision_msgs::Detection3DArray detections3d_msg;
    sensor_msgs::PointCloud2 detection_cloud_msg;
    visualization_msgs::MarkerArray object_markers, plane_markers;

    // Header bilgilerini ayarla
    detections3d_msg.header = cloud_msg->header;
    detections3d_msg.header.stamp = yolo_result_msg->header.stamp;

    // Tüm tespitler için birleştirilmiş nokta bulutunu sakla
    pcl::PointCloud<pcl::PointXYZ> combined_detection_cloud;

    // İşlenen tespit sayacı
    int processed_detection_count = 0;

    // Her bir YOLO tespiti için işlem yap
    for (size_t i = 0; i < yolo_result_msg->detections.detections.size(); i++) {
      const auto& detection = yolo_result_msg->detections.detections[i];

      // Tespit ID'sini kontrol et, atlanacak ID'lerden biriyse sonraki tespite
      // geç
      if (!detection.results.empty()) {
        int detection_id = detection.results[0].id;
        if (skip_detection_ids.find(detection_id) != skip_detection_ids.end()) {
          // ROS_INFO_STREAM("Skipping detection with ID: " << detection_id << "
          // (" << (id_to_prop_name.find(detection_id) != id_to_prop_name.end()
          //? id_to_prop_name[detection_id] : "unknown") << ")");
          continue;  // Bu tespiti atla
        }
      }

      // ROI filtresi uygula ve tespit noktalarını al
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);

      if (yolo_result_msg->masks.empty()) {
        // Maske yoksa bounding box kullan
        processPointsWithBbox(downsampled_cloud, detection, detection_cloud);
      } else {
        // Maske varsa onu kullan
        // processPointsWithMask(cloud, yolo_result_msg->masks[i],
        // detection_cloud); NOT: Şimdilik maske işlemeyi atlıyoruz
        continue;
      }

      if (detection_cloud->points.empty()) {
        continue;
      }

      // Tespit ID'si ve prop ismini burada belirleyelim, bbox filtrelemeden
      // sonra
      int detection_id = -1;
      std::string base_frame_id = "object";  // Varsayılan isim

      // Eğer tespit sonuçları varsa, tespitin sınıf ID'sini doğrudan kullan
      if (!detection.results.empty()) {
        // Tespitin ilk (veya tek) sonucunu al
        detection_id = detection.results[0].id;

        // ID'yi prop ismine dönüştür
        if (detection_id >= 0 &&
            id_to_prop_name.find(detection_id) != id_to_prop_name.end()) {
          base_frame_id = id_to_prop_name[detection_id];
        }
      }

      // Euclidean cluster extraction uygula
      std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters =
          euclideanClusterExtraction(detection_cloud);

      // Eğer hiç küme bulunamadıysa sonraki tespite geç
      if (clusters.empty()) {
        continue;
      }

      // Bulunan küme sayısını ROS_INFO ile yazdır
      ROS_INFO("For detection ID %d (%s) found %zu clusters", detection_id,
               (detection_id >= 0 && id_to_prop_name.find(detection_id) !=
                                         id_to_prop_name.end()
                    ? id_to_prop_name[detection_id].c_str()
                    : "unknown"),
               clusters.size());

      // En yakın cluster'ı bulmak için değişkenler
      size_t closest_cluster_idx = 0;
      float min_squared_distance = std::numeric_limits<float>::max();
      std::vector<Eigen::Vector4f> centroids(clusters.size());
      std::vector<Eigen::Matrix3f> rotation_matrices(clusters.size());
      std::vector<bool> success_flags(clusters.size(), false);

      // Her bir küme için önce planeSegmentation ve PCA işlemini yap,
      // merkezleri hesapla
      for (size_t cluster_idx = 0; cluster_idx < clusters.size();
           cluster_idx++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_cloud =
            clusters[cluster_idx];

        // Küme boşsa atla
        if (cluster_cloud->points.empty()) {
          continue;
        }

        // Düzlem segmentasyonu ve yüzey dönüşümü (PCA) uygula
        success_flags[cluster_idx] =
            planeSegmentationAndPCA(cluster_cloud, centroids[cluster_idx],
                                    rotation_matrices[cluster_idx], tf_buffer_,
                                    camera_optical_frame_, base_link_frame_);

        // En yakın kümeyi bul (centroid norm karesi en küçük olan)
        if (success_flags[cluster_idx]) {
          float squared_distance =
              centroids[cluster_idx][0] * centroids[cluster_idx][0] +
              centroids[cluster_idx][1] * centroids[cluster_idx][1] +
              centroids[cluster_idx][2] * centroids[cluster_idx][2];

          if (squared_distance < min_squared_distance) {
            min_squared_distance = squared_distance;
            closest_cluster_idx = cluster_idx;
          }
        }
      }

      ROS_INFO("Closest cluster index: %zu with squared distance: %f",
               closest_cluster_idx, min_squared_distance);

      // Şimdi tüm kümeleri işle ve tespit oluştur
      for (size_t cluster_idx = 0; cluster_idx < clusters.size();
           cluster_idx++) {
        // Küme boşsa veya PCA başarısız olduysa atla
        if (!success_flags[cluster_idx]) {
          continue;
        }

        // Her küme için boş olmayan benzersiz frame adı
        const std::string base_name = base_frame_id;  // orijinali koru
        std::string frame_id =
            (cluster_idx == closest_cluster_idx)
                ? base_name + "_closest"
                : base_name + "_cluster_" + std::to_string(cluster_idx);

        // 3D tespit mesajı ve marker oluştur (hesaplanmış centroid ve rotation
        // matrix kullanılır)
        createAndPublishDetection(
            detections3d_msg, object_markers, plane_markers,
            clusters[cluster_idx], centroids[cluster_idx],
            rotation_matrices[cluster_idx], detection.results,
            cloud_msg->header, callback_interval.toSec(), frame_id);
        processed_detection_count++;

        // Tespit noktalarını birleştir
        combined_detection_cloud += *(clusters[cluster_idx]);
      }
    }

    // İşlenmiş tespit sayısı kontrolü
    if (processed_detection_count == 0) {
      return;
    }

    // Birleştirilmiş nokta bulutunu ROS mesajına dönüştür
    pcl::toROSMsg(combined_detection_cloud, detection_cloud_msg);
    detection_cloud_msg.header = cloud_msg->header;

    // İşlenmiş verileri yayınla
    detection_cloud_pub_.publish(detection_cloud_msg);
  }

  // 2D bounding box kullanarak nokta bulutu işleme
  void processPointsWithBbox(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const vision_msgs::Detection2D& detection,
      pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud) {
    try {
      int points_in_bbox = 0;

      // Algılama kutusunu genişlet (roi_expansion_factor parametresi
      // kullanarak)
      float min_x = detection.bbox.center.x -
                    (detection.bbox.size_x / 2) * roi_expansion_factor_;
      float max_x = detection.bbox.center.x +
                    (detection.bbox.size_x / 2) * roi_expansion_factor_;
      float min_y = detection.bbox.center.y -
                    (detection.bbox.size_y / 2) * roi_expansion_factor_;
      float max_y = detection.bbox.center.y +
                    (detection.bbox.size_y / 2) * roi_expansion_factor_;

      // Nokta bulutundaki her noktayı kontrol et
      for (const auto& point : cloud->points) {
        // NaN kontrolü
        if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) {
          continue;
        }

        // Manuel projeksiyon hesaplaması
        if (point.z <= 0) {
          continue;  // Z değeri negatif veya sıfır olan noktaları atla
        }

        // Kamera parametrelerini al
        const double fx = cam_model_.fx();  // Odak uzaklığı x
        const double fy = cam_model_.fy();  // Odak uzaklığı y
        const double cx = cam_model_.cx();  // Optik merkez x
        const double cy = cam_model_.cy();  // Optik merkez y

        // Manuel projeksiyon hesaplaması
        double inv_z = 1.0 / point.z;
        cv::Point2d uv;
        uv.x = fx * point.x * inv_z + cx;
        uv.y = fy * point.y * inv_z + cy;

        // Projeksiyon sonrası değerleri kontrol et
        if (std::isnan(uv.x) || std::isnan(uv.y)) {
          continue;
        }

        ROS_DEBUG("Point processed successfully");

        // Noktanın ROI içinde olup olmadığını kontrol et
        if (point.z > 0 && uv.x >= min_x && uv.x <= max_x && uv.y >= min_y &&
            uv.y <= max_y) {
          detection_cloud->points.push_back(point);
          points_in_bbox++;
        }
      }

      if (points_in_bbox < 20) {
        ROS_DEBUG("Few points (%d) found in bounding box", points_in_bbox);
      }

    } catch (const std::exception& e) {
      ROS_ERROR_STREAM("Exception in processPointsWithBbox: " << e.what());
      detection_cloud->points.clear();
    }
  }

  // Euclidean Cluster Extraction - tüm kümeleri döndüren versiyon
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> euclideanClusterExtraction(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;

    // Çok az nokta varsa kümelemeyi atla ve direkt olarak mevcut noktaları tek
    // küme olarak döndür
    if (cloud->points.size() < min_cluster_size_ || cloud->points.size() < 20) {
      clusters.push_back(cloud);
      return clusters;
    }

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
        new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    // Kümeleme parametrelerini ayarla
    ec.setClusterTolerance(
        cluster_tolerance_);  // Noktalar arası maksimum mesafe
    // Dinamik min_cluster_size kullan - eldeki noktaların en az yarısını
    // içermeli
    ec.setMinClusterSize(std::min(min_cluster_size_,
                                  static_cast<int>(cloud->points.size() / 2)));
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    // Hiç küme bulunamazsa tüm noktaları tek küme olarak döndür
    if (cluster_indices.empty()) {
      clusters.push_back(cloud);
      return clusters;
    }

    // Tüm kümeleri oluştur ve döndür
    for (const auto& indices : cluster_indices) {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(
          new pcl::PointCloud<pcl::PointXYZ>);

      // Kümedeki noktaları al
      for (const auto& idx : indices.indices) {
        cloud_cluster->push_back((*cloud)[idx]);
      }

      // Küme merkezini hesapla (bilgi amaçlı)
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud_cluster, centroid);
      float distance = centroid.norm();  // Merkezden uzaklık

      // Kümeyi listeye ekle
      clusters.push_back(cloud_cluster);
    }

    return clusters;
  }

  // -----------------------------------------------------------------------------
  //  Düzlem segmentasyonu + PCA
  // -----------------------------------------------------------------------------
  bool planeSegmentationAndPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                               Eigen::Vector4f& centroid,
                               Eigen::Matrix3f& rotation_matrix,
                               const tf2_ros::Buffer& tf_buffer,
                               const std::string& camera_optical_frame_,
                               const std::string& base_link_frame_) {
    /* ------------------------------------------------------------------------
     */
    /* 0) ÖN KONTROLLER */
    /* ------------------------------------------------------------------------
     */
    if (cloud->empty()) return false;

    /* ------------------------------------------------------------------------
     */
    /* 1) CENTROID */
    /* ------------------------------------------------------------------------
     */
    pcl::compute3DCentroid(*cloud, centroid);

    if (cloud->size() < 10) {  // nokta az → kimlik yok
      rotation_matrix.setIdentity();
      return true;
    }

    /* ------------------------------------------------------------------------
     */
    /* 2) PCA (yedek) */
    /* ------------------------------------------------------------------------
     */
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    const Eigen::Matrix3f pca_eig = pca.getEigenVectors();

    /* ------------------------------------------------------------------------
     */
    /* 3) RANSAC ile düzlem */
    /* ------------------------------------------------------------------------
     */
    pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01);  // 1 cm
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coeff);

    if (inliers->indices.size() < 5) {  // başarısız → PCA
      rotation_matrix = pca_eig;
      return true;
    }

    /* ------------------------------------------------------------------------
     */
    /* 4) Kameradan bakacak Y‐ekseni (düzlem normali) */
    Eigen::Vector3f n(coeff->values[0], coeff->values[1], coeff->values[2]);
    n.normalize();

    Eigen::Vector3f c_vec(centroid[0], centroid[1], centroid[2]);
    const Eigen::Vector3f y_axis = (-n.dot(c_vec) < n.dot(c_vec)) ? -n : n;

    /* ------------------------------------------------------------------------
     */
    /* 5) İlk kaba Z‐ekseni seçimi (y’ye dik) */
    Eigen::Vector3f z_axis, ref = (std::abs(y_axis.x()) < 0.9f)
                                      ? Eigen::Vector3f::UnitX()
                                      : Eigen::Vector3f::UnitY();
    z_axis = ref.cross(y_axis).normalized();
    Eigen::Vector3f x_axis = y_axis.cross(z_axis).normalized();

    try {
      geometry_msgs::TransformStamped tf_msg =
          tf_buffer.lookupTransform(camera_optical_frame_, base_link_frame_,
                                    ros::Time(0), ros::Duration(0.05));
      tf2::Quaternion q(
          tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
          tf_msg.transform.rotation.z, tf_msg.transform.rotation.w);
      tf2::Matrix3x3 m(q);
      Eigen::Vector3f base_z_cam(m[0][2], m[1][2], m[2][2]);
      base_z_cam.normalize();
      if (z_axis.dot(base_z_cam) < 0.0f) {
        z_axis = -z_axis;
        x_axis = -x_axis;
      }
    } catch (const tf2::TransformException& ex) {
      ROS_WARN_STREAM_THROTTLE(5.0, "TF lookup failed: " << ex.what());
    }

    rotation_matrix.col(0) = x_axis;
    rotation_matrix.col(1) = y_axis;
    rotation_matrix.col(2) = z_axis;
    if (rotation_matrix.determinant() < 0.0f) rotation_matrix.col(2) *= -1.0f;
    return true;
  }

  // 3D tespit mesajı ve marker oluştur
  void createAndPublishDetection(
      vision_msgs::Detection3DArray& detections3d_msg,
      visualization_msgs::MarkerArray& object_markers,
      visualization_msgs::MarkerArray& plane_markers,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
      const Eigen::Vector4f& centroid, const Eigen::Matrix3f& rotation_matrix,
      const std::vector<vision_msgs::ObjectHypothesisWithPose>& results,
      const std_msgs::Header& header, const double& duration,
      const std::string& frame_id) {
    if (cloud->points.empty()) {
      return;
    }
    Eigen::Matrix3f rot = rotation_matrix;  // Kopya oluştur
    Eigen::Quaternionf q(rot);
    // TransformStamped mesajını oluştur ve object_transform_updates topic'ine
    // yayınla
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = header.stamp;
    transform_msg.header.frame_id =
        camera_optical_frame_;                // Point cloud'un frame'i
    transform_msg.child_frame_id = frame_id;  // Cluster/nesne ID'si

    // Pozisyon bilgisini ayarla - hesaplandığı gibi kullan
    transform_msg.transform.translation.x = centroid[0];
    transform_msg.transform.translation.y = centroid[1];
    transform_msg.transform.translation.z = centroid[2];

    // Oryantasyon bilgisini ayarla
    transform_msg.transform.rotation.x = q.x();
    transform_msg.transform.rotation.y = q.y();
    transform_msg.transform.rotation.z = q.z();
    transform_msg.transform.rotation.w = q.w();

    // TransformStamped mesajını yayınla

    // TransformStamped mesajını yayınla
    object_transform_pub_.publish(transform_msg);
  }
  // Nokta bulutunu downsample et
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampleCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // Nokta sayısı çok az ise dönüştürmeyi atla
    if (cloud->points.size() < 100) {
      return cloud;
    }

    // Voxel Grid oluştur
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_,
                           voxel_leaf_size_);  // m cinsinden
    voxel_grid.filter(*downsampled_cloud);

    return downsampled_cloud;
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "process_tracker_with_cloud");
  ProcessTrackerWithCloud tracker;
  ros::spin();
  return 0;
}

/**
 * auv_mapping
 * ROS node for processing PointCloud2 data with YOLO detections
 * Inspired by ultralytics_ros tracker_with_cloud_node
 */

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_geometry/pinhole_camera_model.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <vision_msgs/Detection3DArray.h>
#include <ultralytics_ros/YoloResult.h>  // vision_msgs/Detection2DArray yerine

// PCL Kütüphaneleri
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>  // getMinMax3D için gerekli
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
  ros::Publisher detection3d_pub_;
  ros::Publisher object_marker_pub_;
  ros::Publisher plane_marker_pub_;
  ros::Publisher object_transform_pub_;
  
  // Abonelikler ve senkronizasyon
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub_;
  message_filters::Subscriber<sensor_msgs::PointCloud2> lidar_sub_;
  message_filters::Subscriber<ultralytics_ros::YoloResult> yolo_result_sub_;
  
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::CameraInfo, sensor_msgs::PointCloud2, ultralytics_ros::YoloResult> SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  boost::shared_ptr<Sync> sync_;
  
  // TF
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  
  // Kamera modeli
  image_geometry::PinholeCameraModel cam_model_;
  
  // Son işlem zamanı (marker ömrü için)
  ros::Time last_call_time_;

  // Atlanacak tespit ID'lerini saklayan set
  std::set<int> skip_detection_ids;

  // Tespit ID'sinden prop ismini almak için bir mapping ekleyelim
  std::map<int, std::string> id_to_prop_name = {
      {8, "red_buoy_link"},
      {7, "path_link"},
      {9, "bin_whole_link"},
      {12, "torpedo_map_link"},
      {13, "torpedo_hole_link"},
      {1, "gate_left_link"},
      {2, "gate_right_link"},
      {3, "gate_blue_arrow_link"},
      {4, "gate_red_arrow_link"},
      {5, "gate_middle_part_link"},
      {14, "octagon_link"}
  };

public:
  ProcessTrackerWithCloud() : pnh_("~") {
    // Parametreleri yükle
    pnh_.param<std::string>("camera_info_topic", camera_info_topic_, "camera_info");
    pnh_.param<std::string>("lidar_topic", lidar_topic_, "points_raw");
    pnh_.param<std::string>("yolo_result_topic", yolo_result_topic_, "yolo_result");
    pnh_.param<std::string>("yolo_3d_result_topic", yolo_3d_result_topic_, "yolo_3d_result");
    pnh_.param<float>("cluster_tolerance", cluster_tolerance_, 0.3);
    pnh_.param<float>("voxel_leaf_size", voxel_leaf_size_, 0.1);
    pnh_.param<int>("min_cluster_size", min_cluster_size_, 100);
    pnh_.param<int>("max_cluster_size", max_cluster_size_, 10000);
    pnh_.param<float>("roi_expansion_factor", roi_expansion_factor_, 1.1); // %10 genişletme
    
    // Atlanacak tespit ID'lerini parametre olarak al (varsayılan olarak 7 ve 13 ID'leri atlanır)
    std::vector<int> skip_ids;
    pnh_.getParam("skip_detection_ids", skip_ids);
    if (skip_ids.empty()) {
      // Varsayılan değerleri ayarla
      skip_detection_ids = {}; // path_link ve torpedo_hole_link
    } else {
      // Parametre ile gelen değerleri kullan
      skip_detection_ids.clear();
      skip_detection_ids.insert(skip_ids.begin(), skip_ids.end());
    }
    
    // Yayıncıları başlat
    detection_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("detection_cloud", 1);
    detection3d_pub_ = nh_.advertise<vision_msgs::Detection3DArray>(yolo_3d_result_topic_, 1);
    object_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("object_markers", 1);
    plane_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("plane_markers", 1);
    object_transform_pub_ = nh_.advertise<geometry_msgs::TransformStamped>("/taluy/map/object_transform_updates", 10);
    
    // Abonelikleri başlat
    camera_info_sub_.subscribe(nh_, camera_info_topic_, 10);
    lidar_sub_.subscribe(nh_, lidar_topic_, 10);
    yolo_result_sub_.subscribe(nh_, yolo_result_topic_, 10);
    
    // Senkronizasyonu yapılandır
    sync_ = boost::make_shared<Sync>(SyncPolicy(10), camera_info_sub_, lidar_sub_, yolo_result_sub_);
    sync_->registerCallback(boost::bind(&ProcessTrackerWithCloud::syncCallback, this, _1, _2, _3));
    
    // TF broadcaster oluştur
    tf_broadcaster_.reset(new tf2_ros::TransformBroadcaster());
  }
  
  // Senkronize mesajlar için callback fonksiyonu
  void syncCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg,
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    
    // Nokta bulutu boşsa işlem yapma
    if (cloud->points.empty()) {
      return;
    }
    
    // Nokta bulutunu downsample et
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud = 
        downsampleCloud(cloud);
    
    // İşlenmemiş nokta bulutu ve YOLO sonuçları için 3D tespitleri oluşturacak veri yapılarını hazırla
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
      
      // Tespit ID'sini kontrol et, atlanacak ID'lerden biriyse sonraki tespite geç
      if (!detection.results.empty()) {
        int detection_id = detection.results[0].id;
        if (skip_detection_ids.find(detection_id) != skip_detection_ids.end()) {
          //ROS_INFO_STREAM("Skipping detection with ID: " << detection_id << " (" << 
                        //(id_to_prop_name.find(detection_id) != id_to_prop_name.end() ? 
                          //id_to_prop_name[detection_id] : "unknown") << ")");
          continue; // Bu tespiti atla
        }
      }
      
      // ROI filtresi uygula ve tespit noktalarını al
      pcl::PointCloud<pcl::PointXYZ>::Ptr detection_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      
      if (yolo_result_msg->masks.empty()) {
        // Maske yoksa bounding box kullan
        processPointsWithBbox(downsampled_cloud, detection, detection_cloud);
      } else {
        // Maske varsa onu kullan
        // processPointsWithMask(cloud, yolo_result_msg->masks[i], detection_cloud);
        // NOT: Şimdilik maske işlemeyi atlıyoruz
        continue;
      }
      
      if (detection_cloud->points.empty()) {
        continue;
      }
      
      // Tespit ID'si ve prop ismini burada belirleyelim, bbox filtrelemeden sonra
      int detection_id = -1;
      std::string base_frame_id = "object"; // Varsayılan isim
      
      // Eğer tespit sonuçları varsa, tespitin sınıf ID'sini doğrudan kullan
      if (!detection.results.empty()) {
        // Tespitin ilk (veya tek) sonucunu al
        detection_id = detection.results[0].id;
        
        // ID'yi prop ismine dönüştür
        if (detection_id >= 0 && id_to_prop_name.find(detection_id) != id_to_prop_name.end()) {
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
      ROS_INFO("For detection ID %d (%s) found %zu clusters", 
               detection_id, 
               (detection_id >= 0 && id_to_prop_name.find(detection_id) != id_to_prop_name.end() ? 
                id_to_prop_name[detection_id].c_str() : "unknown"), 
               clusters.size());
      
      // Her bir küme için ayrı işlem yap
      for (size_t cluster_idx = 0; cluster_idx < clusters.size(); cluster_idx++) {
        // Kümeyi al
        pcl::PointCloud<pcl::PointXYZ>::Ptr& cluster_cloud = clusters[cluster_idx];
        
        // Küme boşsa atla
        if (cluster_cloud->points.empty()) {
          continue;
        }
        
        // Bu küme için özgün bir frame_id oluştur
        std::string frame_id = base_frame_id + "_" + std::to_string(cluster_idx);
        
        // Düzlem segmentasyonu ve yüzey dönüşümü (PCA) uygula
        Eigen::Vector4f centroid;
        Eigen::Matrix3f rotation_matrix;
        bool success = planeSegmentationAndPCA(cluster_cloud, centroid, rotation_matrix);
        
        if (success) {
          // 3D tespit mesajı ve marker oluştur
          createAndPublishDetection(detections3d_msg, object_markers, plane_markers, 
                                   cluster_cloud, centroid, rotation_matrix, 
                                   detection.results, cloud_msg->header, callback_interval.toSec(), frame_id);
          processed_detection_count++;
        } else {
          continue;
        }
        
        // Tespit noktalarını birleştir
        combined_detection_cloud += *cluster_cloud;
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
    detection3d_pub_.publish(detections3d_msg);
    detection_cloud_pub_.publish(detection_cloud_msg);
    object_marker_pub_.publish(object_markers);
    plane_marker_pub_.publish(plane_markers);
  }
  
  // 2D bounding box kullanarak nokta bulutu işleme
  void processPointsWithBbox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                             const vision_msgs::Detection2D& detection,
                             pcl::PointCloud<pcl::PointXYZ>::Ptr& detection_cloud) {
    int points_in_bbox = 0;
    
    // Algılama kutusunu genişlet (roi_expansion_factor parametresi kullanarak)
    float min_x = detection.bbox.center.x - (detection.bbox.size_x / 2) * roi_expansion_factor_;
    float max_x = detection.bbox.center.x + (detection.bbox.size_x / 2) * roi_expansion_factor_;
    float min_y = detection.bbox.center.y - (detection.bbox.size_y / 2) * roi_expansion_factor_;
    float max_y = detection.bbox.center.y + (detection.bbox.size_y / 2) * roi_expansion_factor_;
    
    // İlk birkaç noktanın projeksiyon bilgilerini göster (debug için)
    int debug_count = 0;
    
    // Nokta bulutundaki her noktayı kontrol et
    for (const auto& point : cloud->points) {
      // Debug için ilk 5 noktanın projeksiyon bilgisini göster
      if (debug_count < 5) {
        cv::Point3d pt_cv(point.x, point.y, point.z);
        cv::Point2d uv = cam_model_.project3dToPixel(pt_cv);
        debug_count++;
      }
      
      // Noktayı görüntü düzlemine yansıt
      cv::Point3d pt_cv(point.x, point.y, point.z);
      cv::Point2d uv = cam_model_.project3dToPixel(pt_cv);
      
      // Noktanın ROI içinde olup olmadığını kontrol et
      if (point.z > 0 && 
          uv.x >= min_x && uv.x <= max_x &&
          uv.y >= min_y && uv.y <= max_y) {
        detection_cloud->points.push_back(point);
        points_in_bbox++;
      }
    }
    
    // Eğer çok az nokta varsa alternatif yaklaşım deneyelim
    if (points_in_bbox < 20 && !cloud->points.empty()) {
      
      // Basit derinlik tabanlı filtreleme
      float min_depth = 0.1;  // Minimum derinlik (m)
      float max_depth = 10.0;  // Maksimum derinlik (m)
      
      detection_cloud->points.clear();
      points_in_bbox = 0;
      
      for (const auto& point : cloud->points) {
        if (point.z > min_depth && point.z < max_depth) {
          detection_cloud->points.push_back(point);
          points_in_bbox++;
        }
      }
    }
  }
  
  // Euclidean Cluster Extraction - tüm kümeleri döndüren versiyon
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
  euclideanClusterExtraction(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    
    // Çok az nokta varsa kümelemeyi atla ve direkt olarak mevcut noktaları tek küme olarak döndür
    if (cloud->points.size() < min_cluster_size_ || cloud->points.size() < 20) {
      clusters.push_back(cloud);
      return clusters;
    }
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

    // Kümeleme parametrelerini ayarla
    ec.setClusterTolerance(cluster_tolerance_); // Noktalar arası maksimum mesafe
    // Dinamik min_cluster_size kullan - eldeki noktaların en az yarısını içermeli
    ec.setMinClusterSize(std::min(min_cluster_size_, static_cast<int>(cloud->points.size() / 2)));
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
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
      
      // Kümedeki noktaları al
      for (const auto& idx : indices.indices) {
        cloud_cluster->push_back((*cloud)[idx]);
      }

      // Küme merkezini hesapla (bilgi amaçlı)
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cloud_cluster, centroid);
      float distance = centroid.norm(); // Merkezden uzaklık

      // Kümeyi listeye ekle
      clusters.push_back(cloud_cluster);
    }

    return clusters;
  }
  
  // Düzlem segmentasyonu ve PCA
  bool planeSegmentationAndPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                              Eigen::Vector4f& centroid,
                              Eigen::Matrix3f& rotation_matrix) {
    if (cloud->points.empty()) {
      return false;
    }
    
    // Centroid hesapla
    pcl::compute3DCentroid(*cloud, centroid);
    
    // Nokta sayısı çok azsa basit bir matris oluştur (birim matris - rotasyon yok)
    if (cloud->points.size() < 10) {
      rotation_matrix = Eigen::Matrix3f::Identity();
      return true;
    }
    
    // PCA analizi yap
    pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(cloud);
    
    // Eigenvalue ve eigenvector'leri al
    Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
    Eigen::Vector3f eigen_values = pca.getEigenValues();
    
    // Düzlem segmentasyonu (RANSAC kullanarak)
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.01); // 1cm tolerans
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.size() < 5) {
      ROS_INFO("Plane segmentation failed, using PCA directly");
      // PCA sonuçlarını doğrudan kullan (düzlemsellik garanti değil)
      rotation_matrix = eigen_vectors;
      return true;
    }
    
    // Düzlem normalini al
    Eigen::Vector3f plane_normal(coefficients->values[0], 
                                coefficients->values[1], 
                                coefficients->values[2]);
    plane_normal.normalize();
    
    // Y eksenini düzlem normaline hizala
    Eigen::Vector3f y_axis = plane_normal;
    
    // Normalin pozitif Y yönünde olması için
    if (y_axis(1) < 0) {
      y_axis = -y_axis;
    }
    
    // X ve Z eksenlerini oluştur
    Eigen::Vector3f x_axis, z_axis;
    
    // Y'ye dik bir eksen bul (Z olacak)
    if (fabs(y_axis(2)) < fabs(y_axis(0)) && fabs(y_axis(2)) > fabs(y_axis(1))) {
      z_axis = Eigen::Vector3f(0, 0, 1).cross(y_axis);
    } else {
      z_axis = Eigen::Vector3f(1, 0, 0).cross(y_axis);
    }
    z_axis.normalize();
    
    // Z ekseninin pozitif Z yönünde olması için
    if (z_axis(2) > 0) {
      z_axis = -z_axis;
    }
    
    // X ekseni Y ve Z'ye dik olmalı (sağ el kuralına göre yeniden hesapla)
    x_axis = y_axis.cross(z_axis);
    x_axis.normalize();
    
    // Rotasyon matrisini oluştur
    rotation_matrix.col(0) = x_axis;
    rotation_matrix.col(1) = y_axis;
    rotation_matrix.col(2) = z_axis;
    
    return true;
  }
  
  // 3D tespit mesajı ve marker oluştur
  void createAndPublishDetection(vision_msgs::Detection3DArray& detections3d_msg,
                               visualization_msgs::MarkerArray& object_markers,
                               visualization_msgs::MarkerArray& plane_markers,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                               const Eigen::Vector4f& centroid,
                               const Eigen::Matrix3f& rotation_matrix,
                               const std::vector<vision_msgs::ObjectHypothesisWithPose>& results,
                               const std_msgs::Header& header,
                               const double& duration,
                               const std::string& frame_id) {
    if (cloud->points.empty()) {
      return;
    }
    
    // 3D Bounding Box oluştur
    vision_msgs::Detection3D detection3d;
    detection3d.header = header;
    
    // Min ve max noktalar hesapla
    pcl::PointXYZ min_pt, max_pt;
    if (!cloud->points.empty()) {
      pcl::getMinMax3D(*cloud, min_pt, max_pt);
    } else {
      min_pt = max_pt = pcl::PointXYZ(0, 0, 0);
    }
    
    // Merkez noktayı ayarla
    detection3d.bbox.center.position.x = centroid[0];
    detection3d.bbox.center.position.y = centroid[1];
    detection3d.bbox.center.position.z = centroid[2];
    
    // Rotasyon matrisinden quaternion'a dönüştür
    Eigen::Matrix3f rot = rotation_matrix; // Kopya oluştur
    Eigen::Quaternionf q(rot);
    
    // Quaternion değerlerini ata
    detection3d.bbox.center.orientation.x = q.x();
    detection3d.bbox.center.orientation.y = q.y();
    detection3d.bbox.center.orientation.z = q.z();
    detection3d.bbox.center.orientation.w = q.w();
    
    // Boyutları ayarla (min/max farkları)
    float x_size = max_pt.x - min_pt.x;
    float y_size = max_pt.y - min_pt.y;
    float z_size = max_pt.z - min_pt.z;
    
    // Minimum boyut kontrolü
    const float min_size = 0.05; // 5cm
    detection3d.bbox.size.x = std::max(x_size, min_size);
    detection3d.bbox.size.y = std::max(y_size, min_size);
    detection3d.bbox.size.z = std::max(z_size, min_size);
    
    // Tespit sonuçları ata
    detection3d.results = results;
    
    // Tespit mesajı dizisine ekle
    detections3d_msg.detections.push_back(detection3d);
    
    // Obje için TF yayınla
    publishTransform(header, centroid, q, frame_id);
    
    // TransformStamped mesajını oluştur ve object_transform_updates topic'ine yayınla
    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = header.stamp;
    transform_msg.header.frame_id = "camera_depth_optical_frame"; // Point cloud'un frame'i
    transform_msg.child_frame_id = frame_id; // Cluster/nesne ID'si
    
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
    object_transform_pub_.publish(transform_msg);
    
    // Obje Marker'ı oluştur
    visualization_msgs::Marker object_marker;
    object_marker.header = header;
    object_marker.ns = "objects";
    object_marker.id = detections3d_msg.detections.size() - 1;
    object_marker.type = visualization_msgs::Marker::CUBE;
    object_marker.action = visualization_msgs::Marker::ADD;
    
    // Marker pozisyonu
    object_marker.pose.position.x = centroid[0];
    object_marker.pose.position.y = centroid[1];
    object_marker.pose.position.z = centroid[2];
    
    // Marker yönelimi
    object_marker.pose.orientation.x = q.x();
    object_marker.pose.orientation.y = q.y();
    object_marker.pose.orientation.z = q.z();
    object_marker.pose.orientation.w = q.w();
    
    // Marker boyutu
    object_marker.scale.x = detection3d.bbox.size.x;
    object_marker.scale.y = detection3d.bbox.size.y;
    object_marker.scale.z = detection3d.bbox.size.z;
    
    // Marker rengi (yarı saydam yeşil)
    object_marker.color.r = 0.0;
    object_marker.color.g = 1.0;
    object_marker.color.b = 0.0;
    object_marker.color.a = 0.5;
    
    // Marker ömrü
    object_marker.lifetime = ros::Duration(duration);
    
    // Marker dizisine ekle
    object_markers.markers.push_back(object_marker);
    
    // Düzlem marker'ı oluştur
    visualization_msgs::Marker plane_marker;
    plane_marker.header = header;
    plane_marker.ns = "planes";
    plane_marker.id = detections3d_msg.detections.size() - 1;
    plane_marker.type = visualization_msgs::Marker::CUBE;
    plane_marker.action = visualization_msgs::Marker::ADD;
    
    // Plane marker pozisyonu (aynı)
    plane_marker.pose.position = object_marker.pose.position;
    plane_marker.pose.orientation = object_marker.pose.orientation;
    
    // Plane marker boyutu (z ekseninde daha ince)
    plane_marker.scale.x = detection3d.bbox.size.x;
    plane_marker.scale.y = detection3d.bbox.size.y;
    plane_marker.scale.z = 0.01; // 1cm kalınlık
    
    // Plane marker rengi (yarı saydam mavi)
    plane_marker.color.r = 0.0;
    plane_marker.color.g = 0.0;
    plane_marker.color.b = 1.0;
    plane_marker.color.a = 0.5;
    
    // Plane marker ömrü
    plane_marker.lifetime = ros::Duration(duration);
    
    // Plane marker dizisine ekle
    plane_markers.markers.push_back(plane_marker);
  }
  
  // Transform yayınla
  void publishTransform(const std_msgs::Header& header,
                      const Eigen::Vector4f& centroid,
                      const Eigen::Quaternionf& rotation,
                      const std::string& child_frame) {
    geometry_msgs::TransformStamped transform;
    transform.header = header;
    transform.child_frame_id = child_frame;
    
    // Çeviri (translation)
    transform.transform.translation.x = centroid[0];
    transform.transform.translation.y = centroid[1];
    transform.transform.translation.z = centroid[2];
    
    // Döndürme (rotation)
    transform.transform.rotation.x = rotation.x();
    transform.transform.rotation.y = rotation.y();
    transform.transform.rotation.z = rotation.z();
    transform.transform.rotation.w = rotation.w();
    
    // Transform yayınla
    tf_broadcaster_->sendTransform(transform);
  }

  // Nokta bulutunu downsample et
  pcl::PointCloud<pcl::PointXYZ>::Ptr
  downsampleCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Nokta sayısı çok az ise dönüştürmeyi atla
    if (cloud->points.size() < 100) {
      return cloud;
    }
    
    // Voxel Grid oluştur
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(cloud);
    voxel_grid.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_); // m cinsinden
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

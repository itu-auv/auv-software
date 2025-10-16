# RL Tabanlı Navigasyon Sistemi - Implementasyon Planı

**Proje Özeti:** Mevcut ITU AUV ROS yazılım yığınına, pekiştirmeli öğrenme (RL) tabanlı, 3D dünya algısı ve odometri verilerini kullanan, Gazebo simülasyonu üzerinde eğitilen bir navigasyon ajanının entegrasyonu.

---

## 1. GENEL MİMARİ VE TASARIM KARARLARI

### 1.1 Observation Space (Gözlem Uzayı)

Agent, çevresini ve kendi durumunu birleşik bir gözlem uzayı üzerinden algılayacaktır. Bu, `gym.spaces.Dict` olarak tanımlanacaktır.

**Format:**
```python
observation_space = spaces.Dict({
    # 1. 3D Dünya Algısı (TF-tabanlı)
    'world_grid': spaces.Box(
        low=0, high=1,
        shape=(7, 7, 2, 8),  # 7x7 yatay grid, 2 derinlik katmanı, 8 obje sınıfı
        dtype=np.float32
    ),
    # 2. Odometri (Aracın Kendi Durumu)
    'linear_velocity': spaces.Box(
        low=-1.0, high=1.0, shape=(3,), dtype=np.float32 # [vx, vy, vz] normalize
    ),
    'angular_velocity': spaces.Box(
        low=-1.0, high=1.0, shape=(3,), dtype=np.float32 # [wx, wy, wz] normalize
    ),
    'orientation': spaces.Box(
        low=-1.0, high=1.0, shape=(3,), dtype=np.float32 # [roll, pitch, yaw] normalize
    )
})
```

- **World Grid:** Araç merkezli, 3 boyutlu voxel grid yapısı:
  - **Yatay (X-Y) boyutu:** 7×7 grid, her hücre `0.5m × 0.5m` → toplam `3.5m × 3.5m` alan (araç merkezinde ±1.75m)
  - **Dikey (Z) boyutu:** 2 derinlik katmanı, her katman `1.0m` → toplam `2.0m` derinlik aralığı
  - **Obje Kanalları:** 8 farklı obje sınıfı için ayrı kanallar
  - Hücre değerleri, o bölgedeki objenin varlığını ve mesafeye göre ağırlığını temsil eder (0-1 arası).
- **Odometry:** Aracın anlık hız ve yönelim bilgileri, maksimum değerlere bölünerek `[-1, 1]` aralığına normalize edilecektir.

### 1.2 Action Space (Eylem Uzayı)

Agent'ın kararları, `cmd_vel` topic'ine gönderilecek sürekli (continuous) hareket komutları olacaktır.

- **Format:** `gym.spaces.Box` (Continuous)
- **Boyutlar:** 4-DOF (Degrees of Freedom)
  - `linear_x`: İleri/geri hız `[-0.5, 0.5] m/s`
  - `linear_y`: Sağa/sola hız `[-0.5, 0.5] m/s`
  - `linear_z`: Yukarı/aşağı hız `[-0.3, 0.3] m/s`
  - `angular_z`: Yaw dönüş hızı `[-0.5, 0.5] rad/s`
- **Çıktı:** ROS `geometry_msgs/Twist` mesajı.

### 1.3 Reward Function (Ödül Fonksiyonu)

Ödül fonksiyonu, hedefe yönelik verimli ve güvenli davranışı teşvik edecektir.

```python
# Temel bileşenler:
reward = (
    + distance_to_goal_improvement * w1  # Hedefe yaklaşma
    + orientation_alignment * w2          # Hedefe doğru yönelme
    - collision_penalty * w3              # Çarpışma cezası (Gazebo contact sensor)
    - time_penalty * w4                   # Zaman geçtirme cezası
    + goal_reached_bonus * w5             # Hedefe varma bonusu
    - action_magnitude_penalty * w7       # Yüksek efor/enerji cezası
)
```

---

## 2. MEVCUT KOD TABANI ANALİZİ VE ENTEGRASYON

### 2.1 Kullanılacak Mevcut Sistemler

- **Obje Pozisyonları (`world_grid` için):**
  - `auv_vision/auv_detection/camera_detection_pose_estimator.py`: 2D tespiti 3D pozisyona çevirir.
  - `auv_mapping/object_map_tf_server`: Tespit edilen objelerin TF frame'lerini yayınlar. Bu TF frame'leri `world_grid` oluşturmak için kullanılacaktır.
- **Odometri Verisi:**
  - `auv_navigation/auv_localization/`: IMU, DVL ve basınç sensörlerinden gelen verileri birleştirerek `/odometry` topic'inde yayınlayan EKF düğümü.
- **Kontrol Arayüzü:**
  - `auv_control/auv_control/controller_ros.h`: `cmd_vel` topic'ini dinleyerek alt seviye PID kontrolcülere hedef hızları iletir. RL agent'ımız bu topic'e yayın yapacaktır.
- **Simülasyon Ortamı:**
  - **Gazebo**: ROS entegrasyonu ile fizik simülasyonu. Mevcut `taluy_description` URDF modeli Gazebo'da çalıştırılacaktır.

### 2.2 Entegrasyon Noktaları

1. **Observation Pipeline:**
   - **World Grid:** `WorldGridEncoder` sınıfı, ROS TF ağacını (`/tf` topic'i) dinleyerek `taluy/base_link` ve obje frame'leri (`red_pipe_link`, `gate_shark_link` vb.) arasındaki transformasyonları alacak.
   - **Odometry:** `AUVNavigationEnv` sınıfı, `/odometry` topic'ine subscribe olarak aracın hız ve yönelim verilerini alacak.
2. **Action Pipeline:**
   - `AUVNavigationEnv` sınıfı, agent'tan gelen `action`'ı bir `geometry_msgs/Twist` mesajına dönüştürüp `/cmd_vel` topic'ine publish edecek.
3. **Environment Reset/Goal:**
   - **Gazebo Reset:** `/gazebo/reset_simulation` veya `/gazebo/set_model_state` servisleri kullanılarak simülasyon ve araç pozisyonu resetlenecek.
   - **Simülasyon Zamanı:** `/clock` topic'i kullanılarak simülasyon zamanı takip edilecek (sim time mode).
   - Görev hedefleri Gazebo world'de spawn edilen objeler olacak.

---

## 3. GELİŞTİRME AŞAMALARI (ROADMAP)

### Phase 1: Temel Altyapı Kurulumu

#### 3.1 Yeni Paket: `auv_rl_navigation`
Aşağıdaki yapıya sahip yeni bir ROS paketi oluşturulacak:
```
auv_navigation/auv_rl_navigation/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── config/
│   └── rl_params.yaml
├── launch/
│   ├── rl_agent.launch
│   ├── training.launch
│   └── gazebo_training_world.launch
├── worlds/
│   └── training_pool.world       # Gazebo world dosyası
├── scripts/
│   ├── rl_agent_node.py          # Deployment için
│   └── train_agent.py            # Eğitim için
└── src/auv_rl_navigation/
    ├── __init__.py
    ├── environments/
    │   ├── __init__.py
    │   └── auv_env.py            # Ana Gym environment sınıfı
    ├── observation/
    │   ├── __init__.py
    │   └── world_grid_encoder.py # 3D Grid oluşturucu
    └── agents/
        └── ppo_agent.py          # PPO agent ve policy tanımı
```

#### 3.2 World Grid Encoder Implementasyonu
**Dosya:** `src/auv_rl_navigation/observation/world_grid_encoder.py`
```python
import rospy
import tf2_ros
import numpy as np

class WorldGridEncoder:
    """TF haritasından araç merkezli 3D voxel grid oluşturur."""
    def __init__(self, grid_dim_xy=7, grid_dim_z=2, cell_size_xy=0.5, cell_size_z=1.0, object_frames=None):
        """
        Args:
            grid_dim_xy: Yatay (x-y) grid boyutu (7x7)
            grid_dim_z: Dikey (z) grid boyutu / derinlik katman sayısı (2)
            cell_size_xy: X-Y düzleminde hücre boyutu (0.5m)
            cell_size_z: Z ekseninde katman yüksekliği (1.0m)
            object_frames: Takip edilecek obje frame listesi
        """
        self.grid_dim_xy = grid_dim_xy
        self.grid_dim_z = grid_dim_z
        self.cell_size_xy = cell_size_xy
        self.cell_size_z = cell_size_z

        # Grid'in kapsadığı alan
        self.grid_range_xy = (grid_dim_xy * cell_size_xy) / 2.0  # ±1.75m
        self.grid_range_z = grid_dim_z * cell_size_z  # 2.0m toplam

        self.object_frames = object_frames if object_frames else []
        self.num_classes = len(self.object_frames)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def create_grid(self, base_frame="taluy/base_link"):
        """
        Araç merkezli 3D voxel grid oluşturur.

        Returns:
            np.ndarray: shape (grid_dim_xy, grid_dim_xy, grid_dim_z, num_classes)
                       [y_index, x_index, z_index, object_class]
        """
        grid = np.zeros((self.grid_dim_xy, self.grid_dim_xy, self.grid_dim_z, self.num_classes))

        for obj_idx, obj_frame in enumerate(self.object_frames):
            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame, obj_frame, rospy.Time(0), rospy.Duration(0.1)
                )
                x = transform.transform.translation.x  # İleri/geri
                y = transform.transform.translation.y  # Sol/sağ
                z = transform.transform.translation.z  # Yukarı/aşağı

                # Dünya koordinatlarını grid indekslerine çevir
                grid_x, grid_y, grid_z = self._world_to_grid(x, y, z)

                if self._in_bounds(grid_x, grid_y, grid_z):
                    # 3D mesafeye göre ağırlıklandırma
                    distance_3d = np.sqrt(x**2 + y**2 + z**2)
                    max_distance = np.sqrt(
                        (self.grid_range_xy * 1.5)**2 + self.grid_range_z**2
                    )
                    weight = max(0, 1.0 - distance_3d / max_distance)

                    grid[grid_y, grid_x, grid_z, obj_idx] = weight

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                continue

        return grid

    def _world_to_grid(self, x, y, z):
        """
        Dünya koordinatlarını grid indekslerine çevirir.

        Koordinat sistemi (base_link):
        - X: İleri (+) / Geri (-)
        - Y: Sol (+) / Sağ (-)
        - Z: Yukarı (+) / Aşağı (-)

        Grid indeksleri:
        - grid_x: 0 (sol) → 6 (sağ)
        - grid_y: 0 (uzak/ileri) → 6 (yakın/geri)
        - grid_z: 0 (aşağı) → 1 (yukarı)

        Args:
            x, y, z: base_link frame'indeki pozisyon (metre)

        Returns:
            tuple: (grid_x, grid_y, grid_z) indeksleri
        """
        # X ekseni: ileri pozitif, grid'de yukarıdan aşağıya (0: uzak, 6: yakın)
        grid_y = int((self.grid_range_xy - x) / self.cell_size_xy)

        # Y ekseni: sol pozitif, grid'de soldan sağa (0: sol, 6: sağ)
        grid_x = int((y + self.grid_range_xy) / self.cell_size_xy)

        # Z ekseni: yukarı pozitif, katmanlar (0: aşağı, 1: yukarı)
        # Aracın altındaki katman ve üstündeki katman
        grid_z = int((z + self.cell_size_z) / self.cell_size_z)

        return grid_x, grid_y, grid_z

    def _in_bounds(self, grid_x, grid_y, grid_z):
        """Grid sınırları içinde mi kontrol eder."""
        return (0 <= grid_x < self.grid_dim_xy and
                0 <= grid_y < self.grid_dim_xy and
                0 <= grid_z < self.grid_dim_z)
```

#### 3.3 ROS Gym Environment Implementasyonu (Gazebo Entegrasyonu)
**Dosya:** `src/auv_rl_navigation/environments/auv_env.py`
```python
import gym
import rospy
import numpy as np
from gym import spaces
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState, GetModelState, GetWorldProperties
from gazebo_msgs.msg import ModelState, ContactsState
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from auv_rl_navigation.observation.world_grid_encoder import WorldGridEncoder

class AUVNavigationEnv(gym.Env):
    """AUV navigasyonu için OpenAI Gym ortamı (Gazebo entegrasyonu)."""
    def __init__(self, sim_mode=True, use_sim_time=True):
        super().__init__()

        # Simülasyon modu kontrolü
        self.sim_mode = sim_mode
        self.use_sim_time = use_sim_time

        if self.use_sim_time:
            rospy.set_param('/use_sim_time', True)

        # Observation space: 3D grid + odometry
        self.observation_space = spaces.Dict({
            'world_grid': spaces.Box(
                low=0, high=1, shape=(7, 7, 2, 8), dtype=np.float32
            ),
            'linear_velocity': spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            'angular_velocity': spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            'orientation': spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
        })

        # Action space: [vx, vy, vz, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -0.5, -0.3, -0.5]),
            high=np.array([0.5, 0.5, 0.3, 0.5]),
            dtype=np.float32
        )

        # Obje listesi (Gazebo model isimleri ve TF frame isimleri)
        object_frames = [
            'gate_shark_link',
            'gate_sawfish_link',
            'red_pipe_link',
            'white_pipe_link',
            'red_buoy',
            'torpedo_map_link',
            'bin_whole_link',
            'octagon_link'
        ]

        self.world_grid_encoder = WorldGridEncoder(
            grid_dim_xy=7, grid_dim_z=2,
            cell_size_xy=0.5, cell_size_z=1.0,
            object_frames=object_frames
        )

        # ROS Publishers/Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/odometry', Odometry, self._odom_callback)

        # Gazebo servisleri
        if self.sim_mode:
            rospy.wait_for_service('/gazebo/reset_simulation')
            rospy.wait_for_service('/gazebo/set_model_state')
            rospy.wait_for_service('/gazebo/get_model_state')

            self.reset_sim = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
            self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

            # Çarpışma tespiti için (opsiyonel)
            rospy.Subscriber('/contact_sensor', ContactsState, self._contact_callback)

        # State variables
        self.latest_odom = None
        self.collision_detected = False
        self.max_linear_vel = 0.5
        self.max_angular_vel = 0.5
        self.episode_steps = 0
        self.max_episode_steps = 1000

        # Goal pozisyonu (her episode'da değişebilir)
        self.goal_position = np.array([5.0, 0.0, -1.0])  # [x, y, z]
        self.goal_tolerance = 0.5  # metre

    def _get_observation(self):
        """Mevcut observation'ı oluşturur."""
        # 3D World grid
        world_grid = self.world_grid_encoder.create_grid()

        # Odometry verilerini normalize et
        if self.latest_odom is not None:
            lin_vel = np.array([
                self.latest_odom.twist.twist.linear.x,
                self.latest_odom.twist.twist.linear.y,
                self.latest_odom.twist.twist.linear.z
            ]) / self.max_linear_vel

            ang_vel = np.array([
                self.latest_odom.twist.twist.angular.x,
                self.latest_odom.twist.twist.angular.y,
                self.latest_odom.twist.twist.angular.z
            ]) / self.max_angular_vel

            orientation = self.latest_odom.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion([
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            euler_normalized = np.array([roll, pitch, yaw]) / np.pi
        else:
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
            euler_normalized = np.zeros(3)

        return {
            'world_grid': world_grid,
            'linear_velocity': lin_vel,
            'angular_velocity': ang_vel,
            'orientation': euler_normalized
        }

    def step(self, action):
        """Bir adım ilerler ve yeni state, reward, done döner."""
        self.episode_steps += 1

        # Action'ı Twist mesajına çevir ve yayınla
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.linear.y = float(action[1])
        twist.linear.z = float(action[2])
        twist.angular.z = float(action[3])
        self.cmd_vel_pub.publish(twist)

        # Gazebo simülasyonun ilerlemesini bekle
        if self.use_sim_time:
            # Sim time ile senkronize çalış
            rospy.sleep(0.1)  # 10 Hz (sim time'a göre)
        else:
            rospy.sleep(0.1)

        # Yeni observation al
        obs = self._get_observation()

        # Reward hesapla
        reward = self._calculate_reward()

        # Episode bitişini kontrol et
        done = self._check_done()

        info = {
            'episode_steps': self.episode_steps,
            'collision': self.collision_detected
        }

        return obs, reward, done, info

    def reset(self):
        """Ortamı resetler (Gazebo reset)."""
        # Episode değişkenlerini sıfırla
        self.episode_steps = 0
        self.collision_detected = False

        if self.sim_mode:
            # Gazebo simülasyonunu resetle
            try:
                self.reset_sim()
                rospy.sleep(0.5)  # Simülasyonun stabilize olmasını bekle
            except rospy.ServiceException as e:
                rospy.logerr(f"Gazebo reset failed: {e}")

            # Aracı başlangıç pozisyonuna getir
            self._reset_vehicle_pose()

            # Hedef pozisyonu randomize et (opsiyonel)
            # self._randomize_goal()

        # Durmayı komuta et
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        rospy.sleep(0.5)

        return self._get_observation()

    def _reset_vehicle_pose(self):
        """Aracı başlangıç pozisyonuna getirir."""
        try:
            state_msg = ModelState()
            state_msg.model_name = 'taluy'  # Gazebo model ismi
            state_msg.pose.position.x = 0.0
            state_msg.pose.position.y = 0.0
            state_msg.pose.position.z = -1.0

            # Yönelim (quaternion)
            q = quaternion_from_euler(0, 0, 0)  # roll, pitch, yaw
            state_msg.pose.orientation.x = q[0]
            state_msg.pose.orientation.y = q[1]
            state_msg.pose.orientation.z = q[2]
            state_msg.pose.orientation.w = q[3]

            # Hızları sıfırla
            state_msg.twist.linear.x = 0
            state_msg.twist.linear.y = 0
            state_msg.twist.linear.z = 0
            state_msg.twist.angular.x = 0
            state_msg.twist.angular.y = 0
            state_msg.twist.angular.z = 0

            self.set_model_state(state_msg)

        except rospy.ServiceException as e:
            rospy.logerr(f"Set model state failed: {e}")

    def _odom_callback(self, msg):
        """Odometry mesajlarını alır."""
        self.latest_odom = msg

    def _contact_callback(self, msg):
        """Çarpışma tespiti (Gazebo contact sensor)."""
        if len(msg.states) > 0:
            self.collision_detected = True

    def _calculate_reward(self):
        """Reward fonksiyonu."""
        if self.latest_odom is None:
            return 0.0

        reward = 0.0

        # Mevcut pozisyon
        current_pos = np.array([
            self.latest_odom.pose.pose.position.x,
            self.latest_odom.pose.pose.position.y,
            self.latest_odom.pose.pose.position.z
        ])

        # Hedefe olan mesafe
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)

        # Hedefe yaklaşma ödülü
        if hasattr(self, 'prev_distance'):
            improvement = self.prev_distance - distance_to_goal
            reward += improvement * 10.0  # w1

        self.prev_distance = distance_to_goal

        # Hedefe ulaşma bonusu
        if distance_to_goal < self.goal_tolerance:
            reward += 100.0  # w5

        # Çarpışma cezası
        if self.collision_detected:
            reward -= 50.0  # w3

        # Zaman cezası (her adım)
        reward -= 0.1  # w4

        # Action magnitude penalty (enerji verimliliği)
        # reward -= action_magnitude * w7

        return reward

    def _check_done(self):
        """Episode bitişini kontrol eder."""
        if self.latest_odom is None:
            return False

        # Hedefe ulaşıldı mı?
        current_pos = np.array([
            self.latest_odom.pose.pose.position.x,
            self.latest_odom.pose.pose.position.y,
            self.latest_odom.pose.pose.position.z
        ])
        distance_to_goal = np.linalg.norm(current_pos - self.goal_position)

        if distance_to_goal < self.goal_tolerance:
            return True

        # Çarpışma oldu mu?
        if self.collision_detected:
            return True

        # Maksimum adım sayısı aşıldı mı?
        if self.episode_steps >= self.max_episode_steps:
            return True

        # Sınırların dışına çıkıldı mı?
        if abs(current_pos[0]) > 10 or abs(current_pos[1]) > 10:
            return True

        return False
```

#### 3.4 Gazebo World ve Launch Dosyaları

**Dosya:** `worlds/training_pool.world`
```xml
<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="underwater_training">
    <!-- Fizik ayarları -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Underwater lighting -->
    <scene>
      <ambient>0.2 0.3 0.4 1</ambient>
      <background>0.1 0.2 0.3 1</background>
    </scene>

    <!-- Havuz zemini -->
    <include>
      <uri>model://pool_floor</uri>
    </include>

    <!-- Görev objeleri (gate, buoys, vb.) -->
    <!-- Bunlar dinamik olarak spawn edilebilir -->

  </world>
</sdf>
```

**Dosya:** `launch/gazebo_training_world.launch`
```xml
<launch>
  <!-- Gazebo'yu başlat -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="$(find auv_rl_navigation)/worlds/training_pool.world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>

  <!-- AUV modelini spawn et -->
  <include file="$(find taluy_description)/launch/spawn_taluy.launch">
    <arg name="x" value="0.0"/>
    <arg name="y" value="0.0"/>
    <arg name="z" value="-1.0"/>
  </include>

  <!-- Robot kontrol ve sensörler -->
  <include file="$(find auv_bringup)/launch/nav.launch"/>

</launch>
```

**Dosya:** `launch/training.launch`
```xml
<launch>
  <!-- Gazebo simülasyonu -->
  <include file="$(find auv_rl_navigation)/launch/gazebo_training_world.launch"/>

  <!-- RL training script'i -->
  <node name="rl_trainer" pkg="auv_rl_navigation" type="train_agent.py" output="screen"/>
</launch>
```

### Phase 2: Simülasyonda Eğitim

#### 3.5 RL Agent ve Policy
**Dosya:** `src/auv_rl_navigation/agents/ppo_agent.py`
```python
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gym

class Custom3DGridExtractor(BaseFeaturesExtractor):
    """7x7x2x8 grid için özel CNN feature extractor."""
    # ...existing code...

class AUVAgent:
    def __init__(self, env):
        policy_kwargs = dict(
            features_extractor_class=Custom3DGridExtractor,
            features_extractor_kwargs=dict(),
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )

        self.model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./rl_logs/"
        )
```

#### 3.6 Eğitim Script'i
**Dosya:** `scripts/train_agent.py`
```python
#!/usr/bin/env python3
import rospy
from auv_rl_navigation.environments.auv_env import AUVNavigationEnv
from auv_rl_navigation.agents.ppo_agent import AUVAgent

def main():
    rospy.init_node('rl_trainer')

    # Gazebo simülasyonu ile env oluştur
    env = AUVNavigationEnv(sim_mode=True, use_sim_time=True)
    agent = AUVAgent(env)

    rospy.loginfo("Starting RL training with Gazebo simulation...")

    # Eğitim döngüsü
    agent.model.learn(total_timesteps=1_000_000)

    # Modeli kaydet
    agent.model.save("ppo_auv_navigation")
    rospy.loginfo("Training completed. Model saved.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
```

### Phase 3: Deployment (Gerçek Ortamda Çalıştırma)

#### 3.7 RL Agent Node
**Dosya:** `scripts/rl_agent_node.py`
```python
#!/usr/bin/env python3
import rospy
from stable_baselines3 import PPO
from auv_rl_navigation.environments.auv_env import AUVNavigationEnv

class RLAgentNode:
    """Eğitilmiş modeli yükleyip gerçek zamanlı kararlar alan ROS düğümü."""
    def __init__(self):
        rospy.init_node('rl_agent_node')

        model_path = rospy.get_param('~model_path', 'ppo_auv_navigation.zip')
        self.model = PPO.load(model_path)

        # Gerçek robot için env (sim_mode=False)
        self.obs_helper = AUVNavigationEnv(sim_mode=False, use_sim_time=False)

        self.rate = rospy.Rate(10)  # 10 Hz

    def run(self):
        rospy.loginfo("RL Agent Node started. Running inference...")
        while not rospy.is_shutdown():
            obs = self.obs_helper._get_observation()
            action, _ = self.model.predict(obs, deterministic=True)

            # Aksiyonu uygula (cmd_vel publish)
            self.obs_helper.step(action)

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = RLAgentNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
```

#### 3.8 Launch Dosyası
**Dosya:** `launch/rl_agent.launch`
```xml
<launch>
  <node name="rl_agent_node" pkg="auv_rl_navigation" type="rl_agent_node.py" output="screen">
    <param name="model_path" value="$(find auv_rl_navigation)/models/ppo_auv_navigation.zip"/>
  </node>
</launch>
```

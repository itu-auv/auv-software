#!/usr/bin/env python3
"""
AUV Navigation Gym Environment
OpenAI Gym environment for training AUV navigation with RL
"""

import gym
from gym import spaces
import numpy as np
import rospy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_srvs.srv import Empty
import tf2_ros
from auv_rl_navigation.observation.world_grid_encoder import WorldGridEncoder


class AUVNavEnv(gym.Env):
    """
    OpenAI Gym Environment for AUV Navigation

    Observation Space:
        - 3D Voxel Grid: (10, 10, 2, 8) - vehicle-centric object grid
        - Goal Vector: (3,) - relative position to goal [x, y, z]
        - Vehicle State: (6,) - linear velocity (3) + angular velocity (3)

    Action Space:
        - Continuous: [surge, sway, heave, yaw_rate] in range [-1, 1]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        max_episode_steps=500,
        goal_tolerance=1.0,
        object_frames=None,
        base_frame="taluy/base_link",
        world_frame="world",
        goal_frame="gate",
        cmd_vel_topic="/taluy/cmd_vel",
        odom_topic="/taluy/odom",
        visualize=True,
    ):
        """
        Initialize AUV Navigation Environment.

        Args:
            max_episode_steps (int): Maximum steps per episode
            goal_tolerance (float): Distance to goal for success (meters)
            object_frames (list): List of object frame names to track
            base_frame (str): Robot base frame
            world_frame (str): World reference frame
            goal_frame (str): Goal frame name (e.g., "gate", "target")
            cmd_vel_topic (str): Topic to publish velocity commands
            odom_topic (str): Topic to receive odometry
            visualize (bool): Enable visualization
        """
        super(AUVNavEnv, self).__init__()

        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.base_frame = base_frame
        self.world_frame = world_frame
        self.goal_frame = goal_frame
        self.visualize = visualize

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0

        # State variables
        self.current_pose = None
        self.current_velocity = None
        self.previous_distance_to_goal = None

        # World Grid Encoder
        self.object_frames = object_frames if object_frames else []
        self.grid_encoder = WorldGridEncoder(
            grid_dim_xy=10,
            grid_dim_z=2,
            cell_size_xy=0.7,
            cell_size_z=1.0,
            object_frames=self.object_frames,
            visualize=visualize,
        )

        # Define action space: [surge, sway, heave, yaw_rate]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(10, 10, 2, len(self.object_frames)),
                    dtype=np.float32,
                ),
                "goal_vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),
                "velocity": spaces.Box(
                    low=-10.0, high=10.0, shape=(6,), dtype=np.float32
                ),
            }
        )

        # ROS publishers and subscribers
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self._odom_callback)

        # TF listener for pose tracking
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.loginfo(
            f"AUVNavEnv initialized: max_steps={max_episode_steps}, "
            f"goal_tolerance={goal_tolerance}m, goal_frame={goal_frame}, "
            f"tracking {len(self.object_frames)} objects"
        )

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            observation (dict): Initial observation
        """
        self.current_step = 0
        self.episode_count += 1

        # Stop the robot
        self._publish_action(np.zeros(4))
        rospy.sleep(0.1)

        # Reset tracking variables
        self.previous_distance_to_goal = None

        # Get initial observation
        observation = self._get_observation()

        # Calculate initial distance to goal
        if self.current_pose is not None:
            self.previous_distance_to_goal = self._distance_to_goal()

        rospy.loginfo(f"Episode {self.episode_count} started.")

        return observation

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action (np.ndarray): Action [surge, sway, heave, yaw_rate]

        Returns:
            observation (dict): Current observation
            reward (float): Reward for this step
            done (bool): Whether episode is finished
            info (dict): Additional information
        """
        self.current_step += 1

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Execute action
        self._publish_action(action)

        # Wait for action to take effect
        rospy.sleep(0.1)

        # Get observation
        observation = self._get_observation()

        # Calculate reward and check termination
        reward, done, info = self._compute_reward_and_done()

        return observation, reward, done, info

    def _get_observation(self):
        """
        Get current observation from environment.

        Returns:
            dict: Observation dictionary
        """
        # Get 3D voxel grid
        grid = self.grid_encoder.create_grid(self.base_frame)

        # Get goal vector in base_link frame
        goal_vector = self._get_goal_vector()

        # Get velocity state
        velocity = self._get_velocity_state()

        observation = {"grid": grid, "goal_vector": goal_vector, "velocity": velocity}

        return observation

    def _get_goal_vector(self):
        """
        Get goal position relative to robot base frame.

        Returns:
            np.ndarray: [x, y, z] relative goal position
        """
        try:
            # Get transform from base_link to goal_frame
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.goal_frame, rospy.Time(0), rospy.Duration(0.1)
            )

            # Relative goal position
            rel_x = transform.transform.translation.x
            rel_y = transform.transform.translation.y
            rel_z = transform.transform.translation.z

            return np.array([rel_x, rel_y, rel_z], dtype=np.float32)

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            # If TF fails, return default
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _get_velocity_state(self):
        """
        Get current velocity state.

        Returns:
            np.ndarray: [vx, vy, vz, wx, wy, wz]
        """
        if self.current_velocity is None:
            return np.zeros(6, dtype=np.float32)

        return np.array(
            [
                self.current_velocity.linear.x,
                self.current_velocity.linear.y,
                self.current_velocity.linear.z,
                self.current_velocity.angular.x,
                self.current_velocity.angular.y,
                self.current_velocity.angular.z,
            ],
            dtype=np.float32,
        )

    def _compute_reward_and_done(self):
        """
        Compute reward and check if episode is done.

        Returns:
            tuple: (reward, done, info)
        """
        done = False
        info = {}

        # Calculate distance to goal
        distance = self._distance_to_goal()
        info["distance_to_goal"] = distance

        # Success: reached goal
        if distance < self.goal_tolerance:
            reward = 100.0
            done = True
            info["termination_reason"] = "goal_reached"
            rospy.loginfo(f"Goal reached! Distance: {distance:.2f}m")
            return reward, done, info

        # Progress reward: encourage moving towards goal
        if self.previous_distance_to_goal is not None:
            progress = self.previous_distance_to_goal - distance
            reward = progress * 10.0  # Scale progress reward
        else:
            reward = 0.0

        self.previous_distance_to_goal = distance

        # Timeout: max steps reached
        if self.current_step >= self.max_episode_steps:
            reward += -10.0  # Small penalty for timeout
            done = True
            info["termination_reason"] = "timeout"
            rospy.loginfo(f"Episode timeout. Final distance: {distance:.2f}m")
            return reward, done, info

        # Small time penalty to encourage efficiency
        reward -= 0.1

        return reward, done, info

    def _distance_to_goal(self):
        """
        Calculate Euclidean distance to goal frame.

        Returns:
            float: Distance in meters
        """
        try:
            # Get transform from base_link to goal_frame
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.goal_frame, rospy.Time(0), rospy.Duration(0.1)
            )

            # Calculate 3D distance
            distance = np.sqrt(
                transform.transform.translation.x**2
                + transform.transform.translation.y**2
                + transform.transform.translation.z**2
            )

            return distance

        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            return float("inf")

    def _publish_action(self, action):
        """
        Publish action as velocity command.

        Args:
            action (np.ndarray): [surge, sway, heave, yaw_rate]
        """
        cmd = Twist()

        # Scale actions to reasonable velocity ranges
        cmd.linear.x = float(action[0]) * 1.0  # surge: ±1.0 m/s
        cmd.linear.y = float(action[1]) * 0.5  # sway: ±0.5 m/s
        cmd.linear.z = float(action[2]) * 0.5  # heave: ±0.5 m/s
        cmd.angular.z = float(action[3]) * 0.5  # yaw rate: ±0.5 rad/s

        self.cmd_vel_pub.publish(cmd)

    def _odom_callback(self, msg):
        """Callback for odometry messages."""
        self.current_pose = msg.pose.pose
        self.current_velocity = msg.twist.twist

    def render(self, mode="human"):
        """Render the environment (handled by RViz)."""
        pass

    def close(self):
        """Clean up resources."""
        # Stop the robot
        self._publish_action(np.zeros(4))
        rospy.loginfo("Environment closed")

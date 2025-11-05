#!/usr/bin/env python3
"""
AUV Gym Environment
===================

Gymnasium-compatible environment for AUV reinforcement learning.
Supports multiple task types: ResidualControl, EndToEndControl
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench, Twist, Pose
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion
from typing import Dict, Any, Tuple, Optional

from auv_gym.utils.config_manager import ConfigManager


class AUVEnv(gym.Env):
    """
    AUV Gymnasium Environment.

    Supports two task types:
    1. ResidualControl: RL agent adds corrections to PID controller
    2. EndToEndControl: RL agent directly outputs thruster commands

    Attributes:
        config (ConfigManager): Configuration manager
        observation_space (spaces.Box): Observation space
        action_space (spaces.Box): Action space
        metadata (dict): Environment metadata
    """

    metadata = {"render_modes": ["human"]}  # gymnasium metadata

    def __init__(self, config: ConfigManager):
        """
        Initialize AUV environment.

        Args:
            config: ConfigManager instance with loaded configuration
        """
        super(AUVEnv, self).__init__()

        # Store configuration
        self.config = config
        self.task_type = config.get("task")
        self.action_type = config.get("action_type")

        if not rospy.core.is_initialized():
            rospy.init_node("auv_gym_env", anonymous=True)

        self.ros_namespace = config.get_ros_namespace()
        self.sim_dt = config.get_simulation_dt()  # time step of the simulation

        # Setup ROS connections
        self._setup_ros_connections()

        # Define observation and action spaces
        self._define_spaces()

        # Episode management
        self.current_state = None
        self.desired_state = None
        self.episode_step = 0
        self.max_episode_steps = config.get("max_episode_steps")
        self.last_action_pid = np.zeros(6)  # Initialize with zeros

        # Domain randomization settings
        self.dr_config = config.get("domain_randomization", {})

        rospy.loginfo(f"AUVEnv initialized: task={self.task_type}")

    def _setup_ros_connections(self):
        # Subscribers
        self.odom_sub = rospy.Subscriber(
            f"/{self.ros_namespace}/odom", Odometry, self._odom_callback
        )

        # Subscribe to PID controller's wrench output for residual control mode
        if self.config.is_residual_mode():
            self.pid_wrench_sub = rospy.Subscriber(
                f"/{self.ros_namespace}/pid_wrench", Wrench, self._pid_wrench_callback
            )
            rospy.loginfo(f"Subscribed to PID wrench: /{self.ros_namespace}/pid_wrench")

        # Publishers
        if self.action_type == "wrench":
            self.action_pub = rospy.Publisher(
                f"/{self.ros_namespace}/cmd_wrench", Wrench, queue_size=1
            )
        elif self.action_type == "cmd_vel":
            self.action_pub = rospy.Publisher(
                f"/{self.ros_namespace}/cmd_vel", Twist, queue_size=1
            )

        # Target position publisher (for visualization)
        self.target_pub = rospy.Publisher(
            f"/{self.ros_namespace}/rl_target", Pose, queue_size=1
        )

        # Gazebo services for simulation reset
        try:
            rospy.wait_for_service("/gazebo/set_model_state", timeout=5.0)
            self.set_model_state = rospy.ServiceProxy(
                "/gazebo/set_model_state", SetModelState
            )
            rospy.loginfo("Connected to Gazebo set_model_state service")
        except rospy.ROSException:
            rospy.logwarn(
                "Gazebo set_model_state service not available. Reset functionality will be limited."
            )
            self.set_model_state = None

        # Wait for connections
        rospy.loginfo("Waiting for ROS connections...")
        rospy.sleep(1.0)

    def _odom_callback(self, msg: Odometry):
        """Callback for odometry messages."""
        self.current_state = msg

    def _pid_wrench_callback(self, msg: Wrench):
        """Callback for PID controller's wrench output."""
        self.last_action_pid = np.array(
            [
                msg.force.x,
                msg.force.y,
                msg.force.z,
                msg.torque.x,
                msg.torque.y,
                msg.torque.z,
            ]
        )

    def _define_spaces(self):
        """Define observation and action spaces based on task type."""
        obs_dim = self.config.get_observation_dim()
        action_dim = self.config.get_action_dim()
        low, high = self.config.get_action_bounds()

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,  # observations can be unbounded
            shape=(obs_dim,),  # size of observation vector
            dtype=np.float32,
        )

        # Action space (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=low, high=high, shape=(action_dim,), dtype=np.float32
        )

        rospy.loginfo(f"Spaces defined: obs_dim={obs_dim}, action_dim={action_dim}")

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial observation of new episode
            info: Additional information for daily logging
        """
        super().reset(seed=seed)

        # 1. Sample new target/goal
        self.desired_state = self._sample_target()

        # 2. Apply domain randomization
        if self.config.has_domain_randomization():
            self._apply_domain_randomization()

        # 3. Reset simulation (move AUV to start position)
        self._reset_simulation()

        # 4. Wait for state to update
        rospy.sleep(self.sim_dt)

        # 5. Reset episode variables
        self.episode_step = 0
        self.last_action_pid = np.zeros(6)

        # 6. Get initial observation
        observation = self._get_observation()

        info = {
            "episode_step": self.episode_step,
            "target_position": (
                self.desired_state[:3] if self.desired_state is not None else None
            ),
        }

        rospy.loginfo(f"Environment reset. Target: {info['target_position']}")

        return observation, info

    def step(
        self, action_rl: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action_rl: Action from RL agent (normalized to [-1, 1])

        Returns:
            observation: New observation
            reward: Reward for this step
            terminated: Whether episode terminated successfully
            truncated: Whether episode was truncated (timeout/failure)
            info: Additional information
        """
        # 1. Process action based on task type
        action_total = self._process_action(action_rl)

        # 2. Send action to simulation
        self._send_action_to_sim(action_total)

        # 3. Wait for simulation step
        rospy.sleep(self.sim_dt)

        # 4. Update episode counter
        self.episode_step += 1

        # 5. Get new observation
        observation = self._get_observation()

        # 6. Compute reward
        reward = self._compute_reward(action_rl, action_total)

        # 7. Check termination conditions
        terminated = self._check_goal_reached()
        truncated = self._check_failure_or_timeout()

        # 8. Prepare info dict
        info = {
            "episode_step": self.episode_step,
            "position_error": self._compute_position_error(),
            "velocity_error": self._compute_velocity_error(),
            "action_rl": action_rl.tolist(),
            "action_total": action_total.tolist(),
        }

        if terminated:
            rospy.loginfo("Goal reached!")
        elif truncated:
            rospy.logwarn("Episode truncated")

        return observation, reward, terminated, truncated, info

    def _process_action(self, action_rl: np.ndarray) -> np.ndarray:
        """
        Process RL action based on task type.

        Args:
            action_rl: Raw action from RL agent [-1, 1]

        Returns:
            Processed action to send to simulation
        """
        # Scale action
        action_scaling = np.array(self.config.get("action_scaling"))
        action_rl_scaled = action_rl * action_scaling

        if self.task_type == "ResidualControl":
            # Use PID output from subscription (already stored in self.last_action_pid)
            action_pid = self.last_action_pid

            # Combine: Total = PID + RL
            action_total = action_pid + action_rl_scaled

        else:
            # End-to-end: use RL action directly
            action_total = action_rl_scaled

        return action_total

    def _send_action_to_sim(self, action: np.ndarray):
        """
        Send action to simulation via ROS.

        Args:
            action: Action to send [6-DOF]
        """
        if self.action_type == "wrench":
            msg = Wrench()
            msg.force.x = action[0]
            msg.force.y = action[1]
            msg.force.z = action[2]
            msg.torque.x = action[3]
            msg.torque.y = action[4]
            msg.torque.z = action[5]
            self.action_pub.publish(msg)

        elif self.action_type == "cmd_vel":
            msg = Twist()
            msg.linear.x = action[0]
            msg.linear.y = action[1]
            msg.linear.z = action[2]
            msg.angular.x = action[3]
            msg.angular.y = action[4]
            msg.angular.z = action[5]
            self.action_pub.publish(msg)

    def _get_observation(self) -> np.ndarray:
        """
        Construct observation based on task type.

        Returns:
            Observation array
        """
        if self.current_state is None or self.desired_state is None:
            # Return zeros if not initialized
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        error_pose = self._compute_error()[:6]  # [x, y, z, roll, pitch, yaw]
        error_vel = self._compute_error()[6:12]  # [vx, vy, vz, wx, wy, wz]

        if self.task_type == "ResidualControl":
            # Observation: [error_pose, error_vel, action_pid]
            obs = np.concatenate(
                [
                    error_pose,
                    error_vel,
                    (
                        self.last_action_pid
                        if self.last_action_pid is not None
                        else np.zeros(6)
                    ),
                ]
            )

        else:  # EndToEndControl
            # Observation: [error_pose, error_vel]
            obs = np.concatenate([error_pose, error_vel])

        return obs.astype(np.float32)

    def _compute_reward(self, action_rl: np.ndarray, action_total: np.ndarray) -> float:
        """
        Compute reward based on current state and action.

        Args:
            action_rl: RL action
            action_total: Total action sent to sim

        Returns:
            Reward value
        """
        weights = self.config.get("reward_weights")

        # Compute errors
        pos_error = self._compute_position_error()
        vel_error = self._compute_velocity_error()

        # Position reward (exponential)
        r_pos = weights.get("w_pose", 1.0) * np.exp(-pos_error)

        # Velocity reward (exponential)
        r_vel = weights.get("w_vel", 0.5) * np.exp(-vel_error)

        # Effort penalty
        if self.task_type == "ResidualControl":
            # Only penalize RL effort
            effort = np.sum(action_rl**2)
            r_effort = -weights.get("w_effort_rl", 0.1) * effort
        else:
            effort = np.sum(action_total**2)
            r_effort = -weights.get("w_effort", 0.01) * effort

        # Time penalty
        r_time = weights.get("w_time", -0.01)

        # Goal reached bonus
        goal_tolerance = self.config.get("goal_tolerance", 0.5)
        if pos_error < goal_tolerance:
            r_goal = weights.get("w_goal_reached", 100.0)
        else:
            r_goal = 0.0

        total_reward = r_pos + r_vel + r_effort + r_time + r_goal

        return float(total_reward)

    def _compute_error(self) -> np.ndarray:
        """
        Compute error vector [pose_error, velocity_error].

        Returns:
            Error array [12]: [dx, dy, dz, droll, dpitch, dyaw, dvx, dvy, dvz, dwx, dwy, dwz]
        """
        if self.current_state is None or self.desired_state is None:
            return np.zeros(12)

        # Position error
        current_pos = np.array(
            [
                self.current_state.pose.pose.position.x,
                self.current_state.pose.pose.position.y,
                self.current_state.pose.pose.position.z,
            ]
        )
        desired_pos = self.desired_state[:3]
        pos_error = desired_pos - current_pos

        # Orientation error (simplified - TODO: use proper quaternion error)
        # For now, use zeros
        ori_error = np.zeros(3)

        # Velocity error (assuming desired velocity is zero for setpoint tasks)
        current_vel = np.array(
            [
                self.current_state.twist.twist.linear.x,
                self.current_state.twist.twist.linear.y,
                self.current_state.twist.twist.linear.z,
                self.current_state.twist.twist.angular.x,
                self.current_state.twist.twist.angular.y,
                self.current_state.twist.twist.angular.z,
            ]
        )
        desired_vel = np.zeros(6)  # Setpoint task
        vel_error = desired_vel - current_vel

        error = np.concatenate([pos_error, ori_error, vel_error])
        return error

    def _compute_position_error(self) -> float:
        """Compute Euclidean position error."""
        if self.current_state is None or self.desired_state is None:
            return np.inf

        current_pos = np.array(
            [
                self.current_state.pose.pose.position.x,
                self.current_state.pose.pose.position.y,
                self.current_state.pose.pose.position.z,
            ]
        )
        desired_pos = self.desired_state[:3]

        return float(np.linalg.norm(desired_pos - current_pos))

    def _compute_velocity_error(self) -> float:
        """Compute velocity error norm."""
        if self.current_state is None:
            return np.inf

        current_vel = np.array(
            [
                self.current_state.twist.twist.linear.x,
                self.current_state.twist.twist.linear.y,
                self.current_state.twist.twist.linear.z,
            ]
        )

        return float(np.linalg.norm(current_vel))

    def _check_goal_reached(self) -> bool:
        """Check if goal has been reached."""
        goal_tolerance = self.config.get("goal_tolerance", 0.5)
        pos_error = self._compute_position_error()
        return pos_error < goal_tolerance

    def _check_failure_or_timeout(self) -> bool:
        """Check if episode should be truncated."""
        # Timeout
        if self.episode_step >= self.max_episode_steps:
            return True

        return False

    def _sample_target(self) -> np.ndarray:
        """
        Sample a new target position.

        Returns:
            Target state [x, y, z, roll, pitch, yaw]
        """

        # For now, sample random position
        x = np.random.uniform(-10.0, 10.0)
        y = np.random.uniform(-10.0, 10.0)
        z = np.random.uniform(-3.0, -0.5)  # Underwater
        roll = 0.0
        pitch = 0.0
        yaw = np.random.uniform(-np.pi, np.pi)

        target = np.array([x, y, z, roll, pitch, yaw])

        # Publish for visualization
        self._publish_target_marker(target)

        return target

    def _publish_target_marker(self, target: np.ndarray):
        """Publish target pose for visualization."""
        msg = Pose()
        msg.position.x = target[0]
        msg.position.y = target[1]
        msg.position.z = target[2]

        msg.orientation.w = 1.0

        self.target_pub.publish(msg)

    def _reset_simulation(self):
        """Reset simulation to initial state by teleporting AUV to start position."""
        if self.set_model_state is None:
            rospy.logwarn("Cannot reset simulation: Gazebo service not available")
            rospy.sleep(0.5)
            return

        # Get initial pose from config or use default
        initial_pose = self.config.get(
            "initial_pose",
            {"x": 0.0, "y": 0.0, "z": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        )

        # Create ModelState message
        model_state = ModelState()
        model_state.model_name = self.config.get(
            "model_name", "taluy"
        )  # Robot model name in Gazebo
        model_state.reference_frame = "world"

        # Set pose
        model_state.pose.position.x = initial_pose["x"]
        model_state.pose.position.y = initial_pose["y"]
        model_state.pose.position.z = initial_pose["z"]

        # Convert roll, pitch, yaw to quaternion (simplified - assuming small angles)
        # For now, just use zero orientation (identity quaternion)
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = 0.0
        model_state.pose.orientation.w = 1.0

        # Reset velocities to zero
        model_state.twist.linear.x = 0.0
        model_state.twist.linear.y = 0.0
        model_state.twist.linear.z = 0.0
        model_state.twist.angular.x = 0.0
        model_state.twist.angular.y = 0.0
        model_state.twist.angular.z = 0.0

        try:
            resp = self.set_model_state(model_state)
            if resp.success:
                rospy.loginfo(
                    f"Simulation reset: AUV teleported to initial position {[initial_pose['x'], initial_pose['y'], initial_pose['z']]}"
                )
            else:
                rospy.logwarn(f"Simulation reset failed: {resp.status_message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def _apply_domain_randomization(self):
        """Apply domain randomization to physics and controller."""
        # TODO: Implement domain randomization
        # - Randomize mass, drag, buoyancy (Gazebo services)
        # - Randomize PID gains
        # - Randomize sensor noise
        rospy.loginfo("Applying domain randomization (placeholder)")

    def _get_current_pose(self) -> np.ndarray:
        """Get current pose [x, y, z, roll, pitch, yaw]."""
        if self.current_state is None:
            return np.zeros(6)

        pos = np.array(
            [
                self.current_state.pose.pose.position.x,
                self.current_state.pose.pose.position.y,
                self.current_state.pose.pose.position.z,
            ]
        )

        # Convert quaternion to euler angles
        quaternion = (
            self.current_state.pose.pose.orientation.x,
            self.current_state.pose.pose.orientation.y,
            self.current_state.pose.pose.orientation.z,
            self.current_state.pose.pose.orientation.w,
        )
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        ori = np.array([roll, pitch, yaw])

        return np.concatenate([pos, ori])

    def _get_current_velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz, wx, wy, wz]."""
        if self.current_state is None:
            return np.zeros(6)

        vel = np.array(
            [
                self.current_state.twist.twist.linear.x,
                self.current_state.twist.twist.linear.y,
                self.current_state.twist.twist.linear.z,
                self.current_state.twist.twist.angular.x,
                self.current_state.twist.twist.angular.y,
                self.current_state.twist.twist.angular.z,
            ]
        )

        return vel

    def close(self):
        """Cleanup resources."""
        rospy.loginfo("Closing AUVEnv")
        # Unregister ROS subscribers/publishers if needed

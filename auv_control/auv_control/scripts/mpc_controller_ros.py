#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs

from dynamic_reconfigure.server import Server
from auv_control.cfg import MPCWeightsConfig

from geometry_msgs.msg import (
    WrenchStamped,
    PoseStamped,
    Twist,
)
from nav_msgs.msg import Odometry, Path

from auv_common_lib.control.enable_state import ControlEnableHandler
from auv_controllers.mpc_controller import (
    VehicleParams,
    UnderwaterVehicle6DOF_CasADi,
    AuvNMPC,
)


class MPCControllerROS:
    def __init__(self):
        rospy.init_node("mpc_controller_node")

        self.rate = rospy.get_param("~rate", 20.0)
        self.body_frame = rospy.get_param("~body_frame", "taluy/base_link")
        self.ref_frame = rospy.get_param("~depth_control_reference_frame", "odom")
        self.transform_timeout = rospy.get_param("~transform_timeout", 1.0)
        self.vehicle_params = self._load_vehicle_params()
        self.mpc_params = self._load_mpc_params()

        initial_config = self._sync_mpc_params_to_reconfigure()

        dt = 1.0 / self.rate
        self.model = UnderwaterVehicle6DOF_CasADi(self.vehicle_params, dt=dt)

        self.mpc = AuvNMPC(
            model=self.model,
            N=self.mpc_params["N"],
            Q_pos=self.mpc_params["Q_pos"],
            Q_vel=self.mpc_params["Q_vel"],
            Q_ori=self.mpc_params["Q_ori"],
            R_tau=self.mpc_params["R_tau"],
            QN_pos=self.mpc_params["QN_pos"],
            QN_vel=self.mpc_params["QN_vel"],
            QN_ori=self.mpc_params["QN_ori"],
            tau_min=self.mpc_params["tau_min"],
            tau_max=self.mpc_params["tau_max"],
            nu_min=self.mpc_params["nu_min"],
            nu_max=self.mpc_params["nu_max"],
            # compiled_solver_path=rospy.get_param("~compiled_solver_path", None),
        )

        self.w0 = None

        self.current_state = np.zeros(13)
        self.current_state[6] = 1.0

        self.ref_pose = None
        self.ref_twist = np.zeros(6)

        self.last_pose_time = rospy.Time(0)
        self.last_vel_time = rospy.Time(0)
        self.cmd_timeout = 1.0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # subscribers
        self.sub_odom = rospy.Subscriber(
            "odometry", Odometry, self.odom_callback, queue_size=1
        )
        self.sub_pose = rospy.Subscriber(
            "cmd_pose", PoseStamped, self.cmd_pose_callback, queue_size=1
        )
        self.sub_vel = rospy.Subscriber(
            "cmd_vel", Twist, self.cmd_vel_callback, queue_size=1
        )
        self.control_enable_handler = ControlEnableHandler(1.0)

        # publishers
        self.pub_wrench = rospy.Publisher("wrench", WrenchStamped, queue_size=1)
        self.pub_pred_path = rospy.Publisher("mpc/predicted_path", Path, queue_size=1)
        self.pub_ref_path = rospy.Publisher("mpc/reference_path", Path, queue_size=1)

        self.current_weights = {}
        self.reconfigure_server = Server(MPCWeightsConfig, self.reconfigure_callback)
        self.reconfigure_server.update_configuration(initial_config)

        rospy.loginfo("MPC Controller Node Initialized")

    def reconfigure_callback(self, config, level):
        self.current_weights["q_pos"] = np.array(
            [config.w_pos_x, config.w_pos_y, config.w_pos_z]
        )
        self.current_weights["q_vel"] = np.array(
            [
                config.w_vel_x,
                config.w_vel_y,
                config.w_vel_z,
                config.w_vel_roll,
                config.w_vel_pitch,
                config.w_vel_yaw,
            ]
        )
        self.current_weights["q_ori"] = config.w_ori
        self.current_weights["r_tau"] = np.array(
            [
                config.w_tau_x,
                config.w_tau_y,
                config.w_tau_z,
                config.w_tau_roll,
                config.w_tau_pitch,
                config.w_tau_yaw,
            ]
        )
        self.current_weights["qN_pos"] = np.array(
            [config.wN_pos_x, config.wN_pos_y, config.wN_pos_z]
        )
        self.current_weights["qN_vel"] = np.array(
            [
                config.wN_vel_x,
                config.wN_vel_y,
                config.wN_vel_z,
                config.wN_vel_roll,
                config.wN_vel_pitch,
                config.wN_vel_yaw,
            ]
        )
        self.current_weights["qN_ori"] = config.wN_ori

        rospy.loginfo("MPC Weights Updated")
        return config

    def _sync_mpc_params_to_reconfigure(self):
        # Extract diagonals from the loaded matrices
        q_pos = np.diag(self.mpc_params["Q_pos"])
        q_vel = np.diag(self.mpc_params["Q_vel"])
        r_tau = np.diag(self.mpc_params["R_tau"])
        qN_pos = np.diag(self.mpc_params["QN_pos"])
        qN_vel = np.diag(self.mpc_params["QN_vel"])

        # Scalars
        q_ori = self.mpc_params["Q_ori"]
        qN_ori = self.mpc_params["QN_ori"]

        config = {}

        # Set parameters for dynamic reconfigure to pick up
        config["w_pos_x"] = float(q_pos[0])
        config["w_pos_y"] = float(q_pos[1])
        config["w_pos_z"] = float(q_pos[2])

        config["w_ori"] = float(q_ori)

        config["w_vel_x"] = float(q_vel[0])
        config["w_vel_y"] = float(q_vel[1])
        config["w_vel_z"] = float(q_vel[2])
        config["w_vel_roll"] = float(q_vel[3])
        config["w_vel_pitch"] = float(q_vel[4])
        config["w_vel_yaw"] = float(q_vel[5])

        config["w_tau_x"] = float(r_tau[0])
        config["w_tau_y"] = float(r_tau[1])
        config["w_tau_z"] = float(r_tau[2])
        config["w_tau_roll"] = float(r_tau[3])
        config["w_tau_pitch"] = float(r_tau[4])
        config["w_tau_yaw"] = float(r_tau[5])

        config["wN_pos_x"] = float(qN_pos[0])
        config["wN_pos_y"] = float(qN_pos[1])
        config["wN_pos_z"] = float(qN_pos[2])

        config["wN_ori"] = float(qN_ori)

        config["wN_vel_x"] = float(qN_vel[0])
        config["wN_vel_y"] = float(qN_vel[1])
        config["wN_vel_z"] = float(qN_vel[2])
        config["wN_vel_roll"] = float(qN_vel[3])
        config["wN_vel_pitch"] = float(qN_vel[4])
        config["wN_vel_yaw"] = float(qN_vel[5])

        for k, v in config.items():
            rospy.set_param(f"~{k}", v)

        return config

    def _load_vehicle_params(self) -> VehicleParams:
        # Helper to get numpy array from param list
        def get_np(name, default):
            val = rospy.get_param(name, default)
            return np.array(val, dtype=float)

        # Defaults from test_mpc.py
        p = VehicleParams(
            m=rospy.get_param("~model/m", 27.0),
            rg=get_np("~model/rg", [0.0, 0.0, 0.0]),
            I_g=self._get_mat("~model/I_g", [0.75, 2.245, 2.116], 3),
            MA=self._get_mat("~model/MA", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6),
            DL=self._get_mat("~model/DL", [30.0, 31.0, 89.0, 1.87, 1.87, 1.87], 6),
            DQ=self._get_mat("~model/DQ", [77.0, 155.0, 136.0, 7.98, 7.98, 7.98], 6),
            W=rospy.get_param("~model/W", 264.87),
            B=rospy.get_param("~model/B", 264.87),
            rb=get_np("~model/rb", [0.0, 0.0, 0.2]),
        )
        return p

    def _get_mat(self, param_name, default_diag, size):
        val = rospy.get_param(param_name, default_diag)
        arr = np.array(val, dtype=float)
        if arr.size == size:  # Vector -> Diag
            return np.diag(arr)
        elif arr.size == size * size:  # Full matrix
            return arr.reshape(size, size)
        else:
            rospy.logwarn(
                f"Parameter {param_name} has unexpected size {arr.size}. Using default."
            )
            return np.diag(np.array(default_diag, dtype=float))

    def _load_mpc_params(self):
        # Defaults
        params = {}
        params["N"] = rospy.get_param("~mpc/N", 20)

        def get_np(name, default):
            return np.array(rospy.get_param(name, default), dtype=float)

        def to_diag(name, default, size):
            val = get_np(name, default)
            if val.size == size:
                return np.diag(val)
            return val.reshape(size, size)

        params["Q_pos"] = to_diag("~mpc/Q_pos", [30.0, 50.0, 80.0], 3)
        params["Q_vel"] = to_diag("~mpc/Q_vel", [5.0, 5.0, 8.0, 1.0, 1.0, 1.0], 6)
        params["Q_ori"] = rospy.get_param("~mpc/Q_ori", 30.0)
        params["R_tau"] = to_diag("~mpc/R_tau", [0.01, 0.01, 0.02, 0.05, 0.05, 0.05], 6)

        params["QN_pos"] = to_diag("~mpc/QN_pos", [200.0, 200.0, 300.0], 3)
        params["QN_vel"] = to_diag("~mpc/QN_vel", [30.0, 30.0, 40.0, 5.0, 5.0, 5.0], 6)
        params["QN_ori"] = rospy.get_param("~mpc/QN_ori", 2000.0)

        params["tau_min"] = get_np("~mpc/tau_min", [-50] * 3 + [-50] * 3)
        params["tau_max"] = get_np("~mpc/tau_max", [50] * 3 + [50] * 3)
        params["nu_min"] = get_np("~mpc/nu_min", [-1] * 3 + [-0.5] * 3)
        params["nu_max"] = get_np("~mpc/nu_max", [1] * 3 + [0.5] * 3)

        return params

    def odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular

        self.current_state[0:3] = [p.x, p.y, p.z]
        self.current_state[3:7] = [q.x, q.y, q.z, q.w]
        self.current_state[7:10] = [v.x, v.y, v.z]
        self.current_state[10:13] = [w.x, w.y, w.z]

    def cmd_pose_callback(self, msg: PoseStamped):
        source_frame = msg.header.frame_id
        target_frame = self.ref_frame

        final_pose = msg.pose

        if (
            source_frame
            and source_frame != target_frame
            and source_frame != "/" + target_frame
        ):
            try:
                # wait for transform
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rospy.Time(0),
                    rospy.Duration(self.transform_timeout),
                )
                pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, transform)
                final_pose = pose_transformed.pose
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as e:
                rospy.logwarn(
                    f"Failed to transform pose from {source_frame} to {target_frame}: {e}"
                )
                return

        p = final_pose.position
        q = final_pose.orientation
        self.ref_pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        self.last_pose_time = rospy.Time.now()

    def cmd_vel_callback(self, msg: Twist):
        self.ref_twist = np.array(
            [
                msg.linear.x,
                msg.linear.y,
                msg.linear.z,
                msg.angular.x,
                msg.angular.y,
                msg.angular.z,
            ]
        )
        self.last_vel_time = rospy.Time.now()

    def control_loop(self):
        now = rospy.Time.now()
        is_pose_timeouted = (now - self.last_pose_time).to_sec() > self.cmd_timeout
        is_vel_timeouted = (now - self.last_vel_time).to_sec() > self.cmd_timeout

        if not self.control_enable_handler.is_enabled() or (
            is_pose_timeouted and is_vel_timeouted
        ):
            return

        # ref_pose = self.current_state[0:7].copy()
        ref_twist = np.zeros(6)

        if not is_vel_timeouted:
            ref_twist = self.ref_twist

            if not is_pose_timeouted:
                # oose + velocity control
                ref_pose = self.ref_pose
            else:
                # velocity control only
                ref_pose = self.current_state[0:7].copy()

        elif not is_pose_timeouted:
            # pose control only
            ref_pose = self.ref_pose

        x_ref = np.concatenate([ref_pose, ref_twist])

        # solve mpc
        try:
            u_cmd, X_pred, _ = self.mpc.solve(
                self.current_state, x_ref, w0=self.w0, weights=self.current_weights
            )
            self.w0 = self.mpc._w_last

            wrench_msg = WrenchStamped()
            wrench_msg.header.stamp = now
            wrench_msg.header.frame_id = self.body_frame
            wrench_msg.wrench.force.x = u_cmd[0]
            wrench_msg.wrench.force.y = u_cmd[1]
            wrench_msg.wrench.force.z = u_cmd[2]
            wrench_msg.wrench.torque.x = u_cmd[3]
            wrench_msg.wrench.torque.y = u_cmd[4]
            wrench_msg.wrench.torque.z = u_cmd[5]

            self.pub_wrench.publish(wrench_msg)

            # Publish predicted path
            path_msg = Path()
            path_msg.header.stamp = now
            path_msg.header.frame_id = self.ref_frame

            for k in range(X_pred.shape[1]):
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = X_pred[0, k]
                pose.pose.position.y = X_pred[1, k]
                pose.pose.position.z = X_pred[2, k]
                pose.pose.orientation.x = X_pred[3, k]
                pose.pose.orientation.y = X_pred[4, k]
                pose.pose.orientation.z = X_pred[5, k]
                pose.pose.orientation.w = X_pred[6, k]
                path_msg.poses.append(pose)

            self.pub_pred_path.publish(path_msg)

            # Publish reference path
            ref_path_msg = Path()
            ref_path_msg.header.stamp = now
            ref_path_msg.header.frame_id = self.ref_frame

            for k in range(self.mpc_params["N"] + 1):
                pose = PoseStamped()
                pose.header = ref_path_msg.header
                pose.pose.position.x = x_ref[0]
                pose.pose.position.y = x_ref[1]
                pose.pose.position.z = x_ref[2]
                pose.pose.orientation.x = x_ref[3]
                pose.pose.orientation.y = x_ref[4]
                pose.pose.orientation.z = x_ref[5]
                pose.pose.orientation.w = x_ref[6]
                ref_path_msg.poses.append(pose)

            self.pub_ref_path.publish(ref_path_msg)

        except Exception as e:
            rospy.logerr(f"MPC Solve Failed: {e}")

    def run(self):
        rate = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            self.control_loop()
            rate.sleep()


if __name__ == "__main__":
    try:
        node = MPCControllerROS()
        node.run()
    except rospy.ROSInterruptException:
        pass

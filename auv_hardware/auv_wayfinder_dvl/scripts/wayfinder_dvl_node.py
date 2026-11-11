#!/usr/bin/env python3
import datetime

import dvl.dvl
import rclpy
from auv_common_lib.ros import get_param, Rate
from auv_common_lib import ros
from rclpy.node import Node
from rclpy.duration import Duration
import dvl
import numpy as np
import std_msgs.msg
import geometry_msgs.msg
import std_srvs.srv
import sensor_msgs.msg


def twist_from_vector(vector: np.array) -> geometry_msgs.msg.Twist:
    msg = geometry_msgs.msg.Twist()
    msg.linear.x = vector[0]
    msg.linear.y = vector[1]
    msg.linear.z = vector[2]
    return msg


def twist_stamped_from_vector(
    node,
    vector: np.array,
    frame_id: str,
    covariance: np.array,
) -> geometry_msgs.msg.TwistWithCovarianceStamped:
    msg = geometry_msgs.msg.TwistWithCovarianceStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = frame_id
    msg.twist.covariance = covariance
    msg.twist.twist = twist_from_vector(vector)

    return msg


class WayfinderNode(Node):
    def __init__(self):
        super().__init__("wayfinder_dvl_node")
        self.time_epoch = datetime.datetime(1970, 1, 1)

        rate = get_param(self, "~rate", default=20.0)

        tf_prefix = get_param(self, "~tf_prefix")
        frame_id = get_param(self, "~frame_id")

        self.frame_id = tf_prefix + "/" + frame_id

        self.velocity_covariance = get_param(self, "~velocity_covariance")
        self.position_covariance = get_param(self, "~position_covariance")
        self.sound_speed = get_param(self, "/env/sound_speed")
        port = get_param(self, "~port")
        baudrate = get_param(self, "~baudrate")
        timeout = get_param(self, "~timeout")
        if timeout <= 0.0:
            timeout = np.inf
        self._initialize_dvl(port, baudrate, timeout)

        self.rate = Rate(self, rate)

        self.pub_is_valid = self.create_publisher(std_msgs.msg.Bool, "is_valid", 1)
        self.pub_velocity_raw = self.create_publisher(
            geometry_msgs.msg.Twist, "velocity_raw", 1
        )
        self.pub_velocity_raw_stamped = self.create_publisher(
            geometry_msgs.msg.TwistWithCovarianceStamped, "velocity_raw_stamped", 1
        )

        self.pub_velocity = self.create_publisher(
            geometry_msgs.msg.Twist, "velocity", 1
        )
        self.pub_velocity_stamped = self.create_publisher(
            geometry_msgs.msg.TwistWithCovarianceStamped, "velocity_stamped", 1
        )

        self.pub_altitude = self.create_publisher(std_msgs.msg.Float32, "altitude", 1)

        self.pub_range = self.create_publisher(sensor_msgs.msg.Range, "range", 1)

        self.set_enabled_service = self.create_service(
            std_srvs.srv.SetBool, "enable", self.set_enable_handler
        )
        self.get_logger().info("Wayfinder DVL ready.")
        self.get_logger().info("use enable service to start pinging..")

    def set_enable_handler(
        self, req: std_srvs.srv.SetBool, response
    ) -> std_srvs.srv.SetBool.Response:
        resp = std_srvs.srv.SetBool.Response()
        if req.data:
            self.wayfinder.register_ondata_callback(self.dvl_data_callback)

            resp.success = self.wayfinder.exit_command_mode()
            if not resp.success:
                resp.message = "Failed to start pinging"
                self.get_logger().error(resp.message)
            else:
                resp.message = "Successfully started pinging"
                self.dvl_enabled = True
                self.get_logger().info(resp.message)

        else:
            resp.success = self.wayfinder.enter_command_mode()
            if not resp.success:
                resp.message = "Failed to stop pinging"
                self.get_logger().error(resp.message)
            else:
                self.wayfinder.unregister_all_callbacks()
                resp.message = "Successfully stopped pinging"
                self.dvl_enabled = False
                self.get_logger().info(resp.message)

        return resp

    def _initialize_dvl(self, port: str, baudrate: int, timeout: float):
        self.wayfinder = dvl.dvl.Dvl(port, baudrate)
        self.get_logger().info("Waiting for connection on {}".format(port))

        start_time = self.get_clock().now()
        while not self.wayfinder.is_connected() and (
            (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout
        ):
            self.get_clock().sleep_for(Duration(seconds=0.1))

        timeouted = not (
            (self.get_clock().now() - start_time).nanoseconds / 1e9 < timeout
        )

        if timeouted:
            self.get_logger().error(
                "Couldn't establish connection to wayfinder on {}, for {} secs.".format(
                    port, timeout
                )
            )
            exit()

        # # Stop pinging
        if not self.wayfinder.enter_command_mode():
            self.get_logger().error("Failed to stop pinging")
            exit()

        # Reset to factory defaults (requires Wayfinder to be in 'command mode')
        if not self.wayfinder.reset_to_defaults():
            self.get_logger().error("Failed to reset to factory defaults")
            exit()

        # if self.use_device_time:
        #     # Sync Device Time
        #     dtime = datetime.datetime.fromtimestamp(
        #         self.get_clock().now().nanoseconds / 1e9, datetime.timezone.utc
        #     )
        #     self.wayfinder.set_time(dtime)

        if not self.wayfinder.set_speed_of_sound(self.sound_speed):
            self.get_logger().error("Sound speed setting failed")

        self.get_logger().debug("Sound speed set to {}".format(self.sound_speed))

        if not self.wayfinder.get_system():
            self.get_logger().error("Failed to get system information")
            exit()

        self.get_logger().debug(
            "System Information:\n{}".format(self.wayfinder._system_info)
        )

        if not self.wayfinder.get_setup():
            self.get_logger().error("Failed to get setup information")
            exit()

        self.get_logger().debug(
            "Setup Information:\n{}".format(self.wayfinder._system_setup)
        )

        if not self.wayfinder.get_features():
            self.get_logger().error("Failed to obtain system features information.")

        self.get_logger().debug("Features:\n{}".format(self.wayfinder._system_features))

        if not self.wayfinder.get_components():
            self.get_logger().error("Failed to obtain information.")

        self.get_logger().debug(
            "HW Components:\n{}".format(self.wayfinder._system_components)
        )

        self.get_logger().debug("Obtaining Time Information")
        htime = self.wayfinder.get_time()

        if htime is None:
            self.get_logger().error("Failed to obtain time information.")

        self.get_logger().info("Device Time: {}".format(htime.ctime()))

    def dvl_data_callback(self, data: dvl.system.OutputData, obj):
        del obj

        if data is None:
            self.get_logger().error("No data received from DVL.")
            return

        velocity_vector = np.array([data.vel_x, data.vel_y, data.vel_z])

        bad_beam = np.isnan(velocity_vector).any() or np.isnan(data.mean_range)

        self.pub_is_valid.publish(not bad_beam)

        if bad_beam:
            velocity_vector = np.array([0.0, 0.0, 0.0])
        #     return

        msg = twist_from_vector(velocity_vector)
        self.pub_velocity_raw.publish(msg)

        msg_stamped = twist_stamped_from_vector(
            self, velocity_vector, self.frame_id, self.velocity_covariance
        )

        self.pub_velocity_raw_stamped.publish(msg_stamped)

        self.pub_altitude.publish(std_msgs.msg.Float32(data=data.mean_range))

        range_msg = sensor_msgs.msg.Range()
        range_msg.header.stamp = self.get_clock().now().to_msg()
        range_msg.header.frame_id = self.frame_id
        range_msg.range = data.mean_range
        range_msg.min_range = 0.0
        range_msg.max_range = 100.0
        self.pub_range.publish(range_msg)


def main():
    ros.init()
    node = WayfinderNode()
    ros.spin(node)


if __name__ == "__main__":
    main()

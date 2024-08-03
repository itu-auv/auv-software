#!/usr/bin/env python3
import datetime

import dvl.dvl
import rospy
import dvl
import numpy as np
import std_msgs.msg
import geometry_msgs.msg
import std_srvs.srv
import auv_common_lib.spatial.velocity_transform


def twist_from_vector(vector: np.array) -> geometry_msgs.msg.Twist:
    msg = geometry_msgs.msg.Twist()
    msg.linear.x = vector[0]
    msg.linear.y = vector[1]
    msg.linear.z = vector[2]
    return msg


def twist_stamped_from_vector(
    vector: np.array,
    frame_id: str,
    covariance: np.array,
) -> geometry_msgs.msg.TwistWithCovarianceStamped:
    msg = geometry_msgs.msg.TwistWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.twist.covariance = covariance
    msg.twist.twist = twist_from_vector(vector)

    return msg


class WayfinderNode:
    def __init__(self):
        self.time_epoch = datetime.datetime(1970, 1, 1)

        rate = rospy.get_param("~rate", default=20.0)

        tf_prefix = rospy.get_param("~tf_prefix")
        frame_id = rospy.get_param("~frame_id")
        base_frame_id = rospy.get_param("~frame_id")

        self.frame_id = tf_prefix + "/" + frame_id
        self.base_frame_id = tf_prefix + "/" + base_frame_id

        self.velocity_covariance = rospy.get_param("~velocity_covariance")
        self.position_covariance = rospy.get_param("~position_covariance")
        self.sound_speed = rospy.get_param("/env/sound_speed")
        port = rospy.get_param("~port")
        baudrate = rospy.get_param("~baudrate")
        timeout = rospy.get_param("~timeout")
        if timeout <= 0.0:
            timeout = np.inf
        self._initialize_dvl(port, baudrate, timeout)

        self.rate = rospy.Rate(rate)
        self.imu_spacial_transformer = (
            auv_common_lib.spatial.velocity_transform.IMUSpatialTransformer(
                self.base_frame_id
            )
        )

        self.pub_is_valid = rospy.Publisher("is_valid", std_msgs.msg.Bool, queue_size=1)
        self.pub_velocity_raw = rospy.Publisher(
            "velocity_raw", geometry_msgs.msg.Twist, queue_size=1
        )
        self.pub_velocity_raw_stamped = rospy.Publisher(
            "velocity_raw_stamped",
            geometry_msgs.msg.TwistWithCovarianceStamped,
            queue_size=1,
        )

        self.pub_velocity = rospy.Publisher(
            "velocity", geometry_msgs.msg.Twist, queue_size=1
        )
        self.pub_velocity_stamped = rospy.Publisher(
            "velocity_stamped",
            geometry_msgs.msg.TwistWithCovarianceStamped,
            queue_size=1,
        )

        self.pub_altitude = rospy.Publisher(
            "altitude", std_msgs.msg.Float32, queue_size=1
        )

        self.set_enabled_service = rospy.Service(
            "enable", std_srvs.srv.SetBool, self.set_enable_handler
        )
        rospy.loginfo("Wayfinder DVL ready.")
        rospy.loginfo("use enable service to start pinging..")

    def set_enable_handler(
        self, req: std_srvs.srv.SetBool
    ) -> std_srvs.srv.SetBoolResponse:
        resp = std_srvs.srv.SetBoolResponse()
        if req.data:
            self.wayfinder.register_ondata_callback(self.dvl_data_callback)

            resp.success = self.wayfinder.exit_command_mode()
            if not resp.success:
                resp.message = "Failed to start pinging"
                rospy.logerr(resp.message)
            else:
                resp.message = "Successfully started pinging"
                self.dvl_enabled = True
                rospy.loginfo(resp.message)

        else:
            resp.success = self.wayfinder.enter_command_mode()
            if not resp.success:
                resp.message = "Failed to stop pinging"
                rospy.logerr(resp.message)
            else:
                self.wayfinder.unregister_all_callbacks()
                resp.message = "Successfully stopped pinging"
                self.dvl_enabled = False
                rospy.loginfo(resp.message)

        return resp

    def _initialize_dvl(self, port: str, baudrate: int, timeout: float):
        self.wayfinder = dvl.dvl.Dvl(port, baudrate)
        rospy.loginfo("Waiting for connection on {}".format(port))

        start_time = rospy.Time.now()
        while not self.wayfinder.is_connected() and (
            (rospy.Time.now() - start_time).to_sec() < timeout
        ):
            rospy.sleep(0.1)

        timeouted = not ((rospy.Time.now() - start_time).to_sec() < timeout)

        if timeouted:
            rospy.logerr(
                "Couldn't establish connection to wayfinder on {}, for {} secs.".format(
                    port, timeout
                )
            )
            exit()

        # # Stop pinging
        if not self.wayfinder.enter_command_mode():
            rospy.logerr("Failed to stop pinging")
            exit()

        # Reset to factory defaults (requires Wayfinder to be in 'command mode')
        if not self.wayfinder.reset_to_defaults():
            rospy.logerr("Failed to reset to factory defaults")
            exit()

        # if self.use_device_time:
        #     # Sync Device Time
        #     dtime = datetime.datetime.fromtimestamp(
        #         rospy.Time.now().to_sec(), datetime.timezone.utc
        #     )
        #     self.wayfinder.set_time(dtime)

        if not self.wayfinder.set_speed_of_sound(self.sound_speed):
            rospy.logerr("Sound speed setting failed")

        rospy.logdebug("Sound speed set to {}".format(self.sound_speed))

        if not self.wayfinder.get_system():
            rospy.logerr("Failed to get system information")
            exit()

        rospy.logdebug("System Information:\n{}".format(self.wayfinder._system_info))

        if not self.wayfinder.get_setup():
            rospy.logerr("Failed to get setup information")
            exit()

        rospy.logdebug("Setup Information:\n{}".format(self.wayfinder._system_setup))

        if not self.wayfinder.get_features():
            rospy.logerr("Failed to obtain system features information.")

        rospy.logdebug("Features:\n{}".format(self.wayfinder._system_features))

        if not self.wayfinder.get_components():
            rospy.logerr("Failed to obtain information.")

        rospy.logdebug("HW Components:\n{}".format(self.wayfinder._system_components))

        rospy.logdebug("Obtaining Time Information")
        htime = self.wayfinder.get_time()

        if htime is None:
            rospy.logerr("Failed to obtain time information.")

        rospy.loginfo("Device Time: {}".format(htime.ctime()))

    def dvl_data_callback(self, data: dvl.system.OutputData, obj):
        del obj

        if data is None:
            rospy.logerr("No data received from DVL.")
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
            velocity_vector, self.frame_id, self.velocity_covariance
        )

        self.pub_velocity_raw_stamped.publish(msg_stamped)

        transformed_velocity_vector = (
            self.imu_spacial_transformer.transform_linear_velocity(
                velocity_vector, self.frame_id
            )
        )

        msg = twist_from_vector(transformed_velocity_vector)
        self.pub_velocity.publish(msg)

        msg_stamped = twist_stamped_from_vector(
            transformed_velocity_vector, self.frame_id, self.velocity_covariance
        )
        self.pub_velocity_stamped.publish(msg_stamped)

        self.pub_altitude.publish(std_msgs.msg.Float32(data.mean_range))


if __name__ == "__main__":
    rospy.init_node("wayfinder_dvl_node")
    node = WayfinderNode()
    rospy.spin()

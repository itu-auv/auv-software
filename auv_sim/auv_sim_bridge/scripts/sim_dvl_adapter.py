#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist
from marine_acoustic_msgs.msg import Dvl
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from std_srvs.srv import SetBool


class SimDvlAdapter(Node):
    def __init__(self):
        super().__init__('sim_dvl_adapter')
        self.declare_parameter('namespace', 'taluy')
        namespace = self.get_parameter('namespace').value.strip('/')
        self.velocity_publisher = self.create_publisher(
            Twist, f'/{namespace}/sensors/dvl/velocity_raw', 10
        )
        self.validity_publisher = self.create_publisher(
            Bool, f'/{namespace}/sensors/dvl/is_valid', 10
        )
        self.altitude_publisher = self.create_publisher(
            Float32, f'/{namespace}/sensors/dvl/altitude', 10
        )
        # as in the ROS 1 simulation mock: missions switch the DVL on and off
        self.dvl_enabled = True
        self.enable_service = self.create_service(
            SetBool, f'/{namespace}/sensors/dvl/enable', self.set_dvl_enable
        )
        self.subscription = self.create_subscription(
            Dvl, f'/model/{namespace}/dvl/velocity', self.on_dvl, 10
        )

    def set_dvl_enable(self, request, response):
        self.dvl_enabled = request.data
        response.success = True
        self.get_logger().info(
            'DVL enable set request received. DVL enabled: %s'
            % ('true' if self.dvl_enabled else 'false')
        )
        return response

    def on_dvl(self, message):
        velocity = Twist()
        valid = Bool()
        valid.data = bool(message.beam_velocities_valid) and self.dvl_enabled
        if valid.data:
            velocity.linear.x = message.velocity.x
            velocity.linear.y = message.velocity.y
            velocity.linear.z = message.velocity.z
        self.velocity_publisher.publish(velocity)
        self.validity_publisher.publish(valid)

        # ros_gz maps the native target slant range into Dvl.altitude. The
        # application altitude contract is the mean sensor-Z beam projection.
        # Publish only when all four ranges are available; no fabricated height.
        if (message.beam_ranges_valid and message.num_good_beams == 4
                and all(math.isfinite(value) and value > 0.0 for value in message.range)
                and all(math.isfinite(vector.z) for vector in message.beam_unit_vec)):
            altitude = Float32()
            altitude.data = float(sum(abs(vector.z) * distance for vector, distance
                                      in zip(message.beam_unit_vec, message.range)) / 4.0)
            self.altitude_publisher.publish(altitude)


def main(args=None):
    rclpy.init(args=args)
    node = SimDvlAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

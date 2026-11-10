#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float32


class SimPressureDepthAdapter(Node):
    """Convert subsea pressure to negative-underwater depth as in SimulationMockROS."""

    def __init__(self):
        super().__init__("sim_pressure_depth_adapter")
        self.declare_parameter("namespace", "taluy")
        self.declare_parameter("standard_pressure", 101352.9)
        self.declare_parameter("gravity", 9.81)
        self.declare_parameter("density", 1028.0)
        namespace = self.get_parameter("namespace").value.strip("/")
        self.standard_pressure = self.get_parameter("standard_pressure").value / 1000.0
        self.gravity = self.get_parameter("gravity").value
        self.density = self.get_parameter("density").value
        self.publisher = self.create_publisher(
            Float32, f"/{namespace}/sensors/external_pressure_sensor/depth", 10
        )
        self.subscription = self.create_subscription(
            FluidPressure, f"/model/{namespace}/pressure", self.on_pressure, 10
        )

    def on_pressure(self, message):
        # UUV's old ROS message was in kPa; the Harmonic bridge uses Pa.
        fluid_pressure = message.fluid_pressure / 1000.0
        depth = (
            1000.0
            * (fluid_pressure - self.standard_pressure)
            / (self.gravity * self.density)
        )
        self.publisher.publish(Float32(data=-depth))


def main(args=None):
    rclpy.init(args=args)
    node = SimPressureDepthAdapter()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

"""dynamic_reconfigure's Server and Client, on top of ROS 2 parameters.

A Server is a set of declared parameters on a node; `ros2 param set` and
rqt_reconfigure edit them. A Client reads and writes another node's parameters
and is told when they change.
"""

import threading
import time

import rclpy
from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    ParameterEvent,
    SetParametersResult,
)
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.parameter import Parameter, parameter_value_to_python
from rclpy.parameter_client import AsyncParameterClient


class Config(dict):
    """Parameter values, readable as config["name"] and config.name."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _wait(node, future, timeout=None):
    # Before spin() (a constructor) nobody processes the answer, so spin here. rclpy
    # leaves the global executor set on the node after doing that once.
    if node.executor is None or node.executor is rclpy.get_global_executor():
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    else:
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        done.wait(timeout)
    if not future.done():
        raise TimeoutError("parameter service did not answer")
    return future.result()


class Server:
    def __init__(self, node, parameters, callback):
        """parameters: (name, default, description[, min, max]) tuples, as in a .cfg file."""
        self.callback = callback
        self.config = Config()
        for name, default, description, *limits in parameters:
            descriptor = ParameterDescriptor(description=description)
            if limits and isinstance(default, float):
                descriptor.floating_point_range = [
                    FloatingPointRange(
                        from_value=float(limits[0]), to_value=float(limits[1])
                    )
                ]
            elif limits:
                descriptor.integer_range = [
                    IntegerRange(from_value=limits[0], to_value=limits[1], step=1)
                ]
            self.config[name] = node.declare_parameter(name, default, descriptor).value
        self.config = callback(self.config, 0) or self.config
        self._handle = node.add_on_set_parameters_callback(self._on_set)

    def _on_set(self, parameters):
        config = Config(self.config)
        for parameter in parameters:
            if parameter.name in config:
                config[parameter.name] = parameter.value
        self.config = self.callback(config, 0) or config
        return SetParametersResult(successful=True)


class Client:
    def __init__(self, node, name, timeout=None, config_callback=None):
        self.node = node
        self.config_callback = config_callback
        self.config = Config()
        self.client = AsyncParameterClient(
            node, name, callback_group=ReentrantCallbackGroup()
        )
        if config_callback is not None:
            namespace = node.get_namespace().rstrip("/")
            self.remote = name if name.startswith("/") else f"{namespace}/{name}"
            self._events = node.create_subscription(
                ParameterEvent, "/parameter_events", self._on_event, 100
            )
        if self.client.wait_for_services(timeout_sec=timeout):
            self._connected(timeout)
        else:
            # When a whole system starts at once, discovery can take longer than any
            # timeout that made sense with a ROS 1 master. Keep waiting in the
            # background instead of running without the server for good.
            node.get_logger().warning(
                f"parameter services of {name} not found yet, still waiting"
            )
            threading.Thread(target=self._connect_later, daemon=True).start()

    def _connect_later(self):
        while rclpy.ok() and (
            self.node.executor is None
            or self.node.executor is rclpy.get_global_executor()
            or not self.client.wait_for_services(timeout_sec=1.0)
        ):
            time.sleep(0.2)
        if rclpy.ok():
            self._connected(None)

    def _connected(self, timeout):
        self.config = self.get_configuration(timeout)
        if self.config_callback is not None:
            self.config_callback(self.config)

    def _on_event(self, event):
        if event.node != self.remote:
            return
        for parameter in list(event.new_parameters) + list(event.changed_parameters):
            self.config[parameter.name] = parameter_value_to_python(parameter.value)
        self.config_callback(self.config)

    def get_configuration(self, timeout=None):
        if not self.client.services_are_ready():
            raise TimeoutError("parameter services are not available")
        names = _wait(self.node, self.client.list_parameters(), timeout).result.names
        values = _wait(self.node, self.client.get_parameters(names), timeout).values
        return Config(
            (name, parameter_value_to_python(value))
            for name, value in zip(names, values)
        )

    def update_configuration(self, changes):
        parameters = [Parameter(name, value=value) for name, value in changes.items()]
        results = _wait(self.node, self.client.set_parameters(parameters)).results
        for result in results:
            if not result.successful:
                raise ValueError(result.reason)
        self.config.update(changes)
        return self.config

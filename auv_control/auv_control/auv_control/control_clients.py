import threading
import inspect
import signal

from rclpy.executors import ExternalShutdownException
from rclpy.signals import SignalHandlerOptions

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient


class ControlClients:
    def __init__(self, namespace):
        self.node = Node('control_clients', namespace=namespace, use_global_arguments=False)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        self.services = {}

    def _spin(self):
        try:
            self.executor.spin()
        except ExternalShutdownException:
            pass

    def wait(self, future, timeout=None):
        event = threading.Event()
        future.add_done_callback(lambda _: event.set())
        if not event.wait(timeout):
            raise TimeoutError('ROS service request timed out')
        return future.result()

    def service(self, srv_type, name, request):
        key = (srv_type, name)
        if key not in self.services:
            self.services[key] = self.node.create_client(srv_type, name)
        client = self.services[key]
        if not client.wait_for_service(timeout_sec=0.2):
            raise RuntimeError(f'Service unavailable: {name}')
        return self.wait(client.call_async(request))

    def shutdown(self):
        self.executor.shutdown()
        self.thread.join()
        self.node.destroy_node()


class ControllerParameters:
    def __init__(self, clients, target, timeout=5):
        self.clients = clients
        self.client = AsyncParameterClient(clients.node, target)
        if not self.client.wait_for_services(timeout_sec=timeout):
            raise RuntimeError(f'Parameter services unavailable: {target}')

    def get_parameter(self, name):
        response = self.clients.wait(self.client.get_parameters([name]))
        return rclpy.parameter.parameter_value_to_python(response.values[0])

    def get_configuration(self):
        names = ['kp', 'ki', 'kd', 'integral_clamp_limits', 'max_velocity',
                 'max_acceleration', 'max_acceleration_rate', 'gravity_compensation_z']
        response = self.clients.wait(self.client.get_parameters(names))
        return {name: rclpy.parameter.parameter_value_to_python(value)
                for name, value in zip(names, response.values)}

    def update_configuration(self, values):
        params = [Parameter(name, value=([float(item) for item in value]
                                        if isinstance(value, (list, tuple)) else float(value)))
                  for name, value in values.items()]
        response = self.clients.wait(self.client.set_parameters_atomically(params))
        if not response.result.successful:
            raise ValueError(response.result.reason)
        return self.get_configuration()


def log(node, level, message, *args, throttle=None):
    if args:
        message = message % args
    frame = inspect.currentframe().f_back
    if frame.f_code.co_name == 'log_throttle':
        frame = frame.f_back
    key = (frame.f_code.co_filename, frame.f_lineno)
    if not hasattr(node, '_control_loggers'):
        node._control_loggers = {}
    if key not in node._control_loggers:
        node._control_loggers[key] = rclpy.logging.get_logger(node.get_logger().name)
    method = getattr(node._control_loggers[key], level)
    if throttle is None:
        method(message)
    else:
        method(message, throttle_duration_sec=throttle)


def parameter(node, name, default):
    name = name.removeprefix('~').replace('/', '.')
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


def log_throttle(node, level, seconds, message, *args):
    log(node, level, message, *args, throttle=seconds)


def init_ros():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    def stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)

import inspect
import os
import signal
import threading
import time

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions

_UNSET = object()


def _param_name(name):
    return name.lstrip("~/").replace("/", ".")


def _nested_overrides(node, name):
    """Parameter overrides below `name` as a nested dict (rospy returned dicts)."""
    nested = {}
    for key, parameter in getattr(node, "_parameter_overrides", {}).items():
        if key.startswith(name + "."):
            branch = nested
            *parents, leaf = key[len(name) + 1 :].split(".")
            for parent in parents:
                branch = branch.setdefault(parent, {})
            branch[leaf] = parameter.value
    return nested


def get_param(node, name, default=_UNSET):
    """rospy.get_param for a node's own parameters; `~x`, `x/y` and `/x/y` all map to `x.y`."""
    name = _param_name(name)
    nested = _nested_overrides(node, name)
    if nested or isinstance(default, dict):
        return nested or default
    if not node.has_parameter(name):
        node.declare_parameter(
            name,
            None if default is _UNSET else default,
            ParameterDescriptor(dynamic_typing=True),
        )
    value = node.get_parameter(name).value
    if value is None:
        if default is _UNSET:
            raise KeyError(name)
        return default
    return value


def has_param(node, name):
    return get_param(node, name, None) is not None


def latched_qos(depth=1):
    """Equivalent of a ROS 1 latched publisher; late subscribers need the same profile."""
    return QoSProfile(depth=depth, durability=DurabilityPolicy.TRANSIENT_LOCAL)


_node = None


def set_node(node):
    """Register the process's node for code that has no node at hand (see get_node)."""
    global _node
    _node = node


def get_node():
    """The node registered with set_node: for helper modules and SMACH states that
    used rospy's implicit, process-wide node."""
    if _node is None:
        raise RuntimeError("auv_common_lib.ros.set_node() was not called")
    return _node


def init_node(name):
    """rospy.init_node for code written around rospy's process-wide node: init(),
    create the node and register it for get_node()."""
    init()
    node = rclpy.create_node(name)
    set_node(node)
    return node


class ROSException(Exception):
    pass


class ServiceException(Exception):
    pass


class ServiceUnavailable(ROSException, ServiceException):
    pass


class ServiceProxy:
    """rospy.ServiceProxy on an rclpy client.

    Callable and blocking, also from inside callbacks (see spin). Like rospy, and
    unlike a bare rclpy client, calling a service nobody serves raises instead of
    waiting forever.
    """

    def __init__(self, node, srv_type, name):
        self.node = node
        self.srv_type = srv_type
        self.created = time.monotonic()
        self.client = node.create_client(
            srv_type, name, callback_group=ReentrantCallbackGroup()
        )

    def wait_for_service(self, timeout=None):
        if not self.client.wait_for_service(timeout_sec=timeout):
            raise ServiceUnavailable(f"service {self.client.srv_name} is not available")

    def call(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], self.srv_type.Request):
            request = args[0]
        else:
            fields = self.srv_type.Request.get_fields_and_field_types()
            request = self.srv_type.Request(**dict(zip(fields, args)), **kwargs)
        if not self.client.service_is_ready():
            # a new client needs a few seconds to discover the server in a large
            # system; after that a missing server fails at once, as it did with rospy
            self.wait_for_service(
                timeout=max(0.0, 5.0 - (time.monotonic() - self.created))
            )
        executor = self.node.executor
        if executor is None or executor is rclpy.get_global_executor():
            # called before spin() (a constructor): nobody else processes the answer
            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            return future.result()
        return self.client.call(request)

    __call__ = call


def service_client(node, srv_type, name):
    """The node's ServiceProxy for a service; created on first use."""
    proxies = node.__dict__.setdefault("_auv_service_proxies", {})
    if (srv_type, name) not in proxies:
        proxies[srv_type, name] = ServiceProxy(node, srv_type, name)
    return proxies[srv_type, name]


def wait_for_service(proxy, timeout=None):
    """rospy.wait_for_service: raises ServiceUnavailable on timeout."""
    proxy.wait_for_service(timeout)


class Rate:
    """rospy.Rate. Unlike node.create_rate() it owns no timer, so making one per
    state execution or per call leaks nothing."""

    def __init__(self, node, hz):
        self.clock = node.get_clock()
        self.period = Duration(seconds=1.0 / hz)
        self.deadline = self.clock.now() + self.period

    def sleep(self):
        if not rclpy.ok() or not self.clock.sleep_until(self.deadline):
            if not rclpy.ok():
                raise ROSInterruptException()
        now = self.clock.now()
        self.deadline = self.deadline + self.period
        if self.deadline < now:  # fell behind (or time jumped): start over from now
            self.deadline = now + self.period


def wait_for_message(node, topic, msg_type, timeout=None, latched=False):
    """rospy.wait_for_message: raises ROSException on timeout."""
    from rclpy.wait_for_message import wait_for_message as wait

    received, msg = wait(
        msg_type,
        node,
        topic,
        qos_profile=latched_qos() if latched else 1,
        time_to_wait=-1 if timeout is None else timeout,
    )
    if not received:
        raise ROSException(f"timeout waiting for a message on {topic}")
    return msg


_shutdown_hooks = []


def on_shutdown(callback):
    """rospy.on_shutdown: `callback` runs when spin() ends, while the node can still
    publish and call services."""
    _shutdown_hooks.append(callback)


def sim_time_arg(node):
    """Launch argument that gives a child launch the same clock as this node."""
    return "use_sim_time:=%s" % str(node.get_parameter("use_sim_time").value).lower()


def _refuse_mixed_clocks(node):
    """Stop a node that runs on wall time next to a simulation.

    ROS 1 had one /use_sim_time for everything; here each launch has to pass it. A
    mission started without it would stamp and time everything on the wrong clock.
    """
    if node.get_parameter("use_sim_time").value:
        return
    deadline = time.monotonic() + 20.0
    while rclpy.ok() and time.monotonic() < deadline:
        if node.count_publishers("/clock"):
            node.get_logger().fatal(
                "A simulation clock is running but this node uses wall time. "
                "Start it with use_sim_time:=true."
            )
            os.kill(os.getpid(), signal.SIGINT)
            return
        time.sleep(0.5)


def spin(node, loop=None):
    """Run a node until shutdown, then clean up.

    Callbacks run on executor threads. `loop`, if given, is the node's rospy-style
    main loop (create_rate, blocking service calls); it runs in the calling thread.
    """
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=_refuse_mixed_clocks, args=(node,), daemon=True).start()

    def spin_executor():
        try:
            executor.spin()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass

    thread = threading.Thread(target=spin_executor, daemon=True)
    try:
        if loop is None:
            spin_executor()
        else:
            thread.start()
            loop()
    except (KeyboardInterrupt, ExternalShutdownException, ROSInterruptException):
        pass
    finally:
        for hook in _shutdown_hooks:
            try:
                hook()
            except Exception as error:
                node.get_logger().error(f"shutdown hook failed: {error}")
        executor.shutdown(timeout_sec=1.0)
        if thread.is_alive():
            thread.join(2.0)  # a still-spinning thread aborts the process at exit
        node.destroy_node()
        try:
            rclpy.try_shutdown()
        except Exception:
            pass


# --- lenient message fields -------------------------------------------------
# rospy accepted an int in a float field, a numpy scalar, or a Vector3 where a
# Point belongs. rclpy aborts the whole process on those when the message is
# serialized. coerce() repairs such fields; init() applies it to everything a
# process publishes, requests or responds with.

_FLOATS = ("float", "double", "long double")
_classes = {}


def _message_class(slot_type):
    key = (*slot_type.namespaces, slot_type.name)
    if key not in _classes:
        import importlib

        module = importlib.import_module(".".join(slot_type.namespaces))
        _classes[key] = getattr(module, slot_type.name)
    return _classes[key]


def _coerce_value(value, slot_type):
    from rosidl_parser.definition import (
        AbstractNestedType,
        BasicType,
        NamespacedType,
    )

    if isinstance(slot_type, BasicType):
        name = slot_type.typename
        if name in _FLOATS:
            return value if isinstance(value, float) else float(value)
        if name == "boolean":
            return value if isinstance(value, bool) else bool(value)
        if name in ("octet", "char", "wchar"):
            return value
        return value if isinstance(value, int) else int(value)
    if isinstance(slot_type, NamespacedType):
        cls = _message_class(slot_type)
        if hasattr(value, "to_msg"):  # rclpy Time / Duration
            value = value.to_msg()
        if type(value) is not cls:
            converted = cls()
            for slot in cls.__slots__:
                if hasattr(value, slot):
                    setattr(converted, slot, getattr(value, slot))
            value = converted
        return coerce(value)
    if isinstance(slot_type, AbstractNestedType) and isinstance(
        slot_type.value_type, NamespacedType
    ):
        return [_coerce_value(item, slot_type.value_type) for item in value]
    return value


def coerce(msg):
    for slot, slot_type in zip(msg.__slots__, msg.SLOT_TYPES):
        value = getattr(msg, slot)
        fixed = _coerce_value(value, slot_type)
        if fixed is not value:
            setattr(msg, slot, fixed)
    return msg


def init(args=None):
    """rclpy.init() plus rospy-like leniency for message field types."""
    from rclpy.client import Client
    from rclpy.publisher import Publisher
    from rclpy.service import Service

    if not getattr(Publisher, "_auv_lenient", False):
        publish, call_async, send_response = (
            Publisher.publish,
            Client.call_async,
            Service.send_response,
        )

        def lenient_publish(self, msg):
            if isinstance(msg, bytes):
                return publish(self, msg)
            if not hasattr(msg, "SLOT_TYPES"):  # rospy: pub.publish(True)
                first_field = next(iter(self.msg_type.get_fields_and_field_types()))
                msg = self.msg_type(**{first_field: msg})
            return publish(self, coerce(msg))

        Publisher.publish = lenient_publish
        Client.call_async = lambda self, request: call_async(self, coerce(request))
        Service.send_response = lambda self, response, header: send_response(
            self, coerce(response), header
        )
        Publisher._auv_lenient = True

    # rospy's TF listener received transforms in a thread of its own from the moment
    # it was created. Give each listener a private node and thread, so lookups also
    # work in constructors and TF traffic does not wake the node's own executor.
    import tf2_ros

    if not getattr(tf2_ros.TransformListener, "_auv_threaded", False):
        listener_init = tf2_ros.TransformListener.__init__

        def threaded_listener(self, buffer, node, *args, spin_thread=None, **kwargs):
            if spin_thread is None:
                spin_thread = True
                node = Node(
                    node.get_name() + "_tf",
                    namespace=node.get_namespace(),
                    use_global_arguments=False,
                    start_parameter_services=False,
                    parameter_overrides=[node.get_parameter("use_sim_time")],
                )
            listener_init(self, buffer, node, *args, spin_thread=spin_thread, **kwargs)

        tf2_ros.TransformListener.__init__ = threaded_listener
        tf2_ros.TransformListener._auv_threaded = True

    # rospy ran every subscriber, timer and service in its own thread. rclpy puts
    # all of a node's callbacks in one mutually exclusive group, so one slow callback
    # stalls the rest. Give each its own group unless the caller chose one.
    if not getattr(Node, "_auv_own_groups", False):
        for method in ("create_subscription", "create_timer", "create_service"):

            def with_own_group(
                self,
                *args,
                _create=getattr(Node, method),
                _signature=inspect.signature(getattr(Node, method)),
                **kwargs,
            ):
                call = _signature.bind(self, *args, **kwargs)
                if call.arguments.get("callback_group") is None:
                    call.arguments["callback_group"] = MutuallyExclusiveCallbackGroup()
                return _create(*call.args, **call.kwargs)

            setattr(Node, method, with_own_group)
        Node._auv_own_groups = True

    # Ctrl-C and SIGTERM end spin() with the node still usable, so shutdown hooks
    # and `finally` blocks can talk to ROS (rclpy's own handlers shut down first).
    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, interrupt)

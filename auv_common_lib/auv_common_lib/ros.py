import threading

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.exceptions import ROSInterruptException
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile

_UNSET = object()


def _param_name(name):
    return name.lstrip("~/").replace("/", ".")


def get_param(node, name, default=_UNSET):
    """rospy.get_param for a node's own parameters; `~x`, `x/y` and `/x/y` all map to `x.y`."""
    name = _param_name(name)
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


class ServiceUnavailable(Exception):
    pass


def service_client(node, srv_type, name):
    """rospy.ServiceProxy: a client whose blocking call() also works from inside
    callbacks (see spin). Asking again for the same service returns the same client."""
    clients = node.__dict__.setdefault("_auv_service_clients", {})
    if (srv_type, name) not in clients:
        clients[srv_type, name] = node.create_client(
            srv_type, name, callback_group=ReentrantCallbackGroup()
        )
    return clients[srv_type, name]


def wait_for_service(client, timeout=None):
    """rospy.wait_for_service: raises ServiceUnavailable on timeout."""
    if not client.wait_for_service(timeout_sec=timeout):
        raise ServiceUnavailable(f"service {client.srv_name} is not available")


def spin(node, loop=None):
    """Run a node until shutdown, then clean up.

    Callbacks run on executor threads. `loop`, if given, is the node's rospy-style
    main loop (create_rate, blocking service calls); it runs in the calling thread.
    """
    executor = MultiThreadedExecutor()
    executor.add_node(node)

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
        Publisher.publish = lambda self, msg: publish(
            self, msg if isinstance(msg, bytes) else coerce(msg)
        )
        Client.call_async = lambda self, request: call_async(self, coerce(request))
        Service.send_response = lambda self, response, header: send_response(
            self, coerce(response), header
        )
        Publisher._auv_lenient = True
    rclpy.init(args=args)

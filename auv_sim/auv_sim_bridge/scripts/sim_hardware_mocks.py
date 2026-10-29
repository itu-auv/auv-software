#!/usr/bin/env python3
"""Ideal simulation board/task mocks. No battery, actuator or acoustic dynamics."""
import math
import json
import copy
from functools import partial
from concurrent.futures import CancelledError
from pathlib import Path
import xml.etree.ElementTree as ET

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ament_index_python.packages import get_package_share_directory
from auv_msgs.srv import SetObjectTransform
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Float32, UInt16, String
from std_srvs.srv import SetBool, Trigger
from ros_gz_interfaces.srv import SpawnEntity, SetEntityPose
from rclpy.clock import Clock, ClockType
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster, TransformException


class SimulationMocks(Node):
    def __init__(self):
        super().__init__('sim_hardware_mocks')
        self.declare_parameter('world', 'pool')
        self.declare_parameter('pinger_mode', 'torpedo')
        self.world = self.get_parameter('world').value
        mode = self.get_parameter('pinger_mode').value
        if mode not in ('torpedo', 'octagon'):
            raise ValueError('pinger_mode must be torpedo or octagon')
        self.target = (5.25, -1.0) if mode == 'torpedo' else (13.0, -4.25)
        self.enabled = False
        self.truth = None
        self.serial = 0
        self.projectiles = {}
        self.pose_pending = {}
        self.dropped = set()
        self.inflight = set()
        self.buffer = Buffer(node=self)
        self.listener = TransformListener(self.buffer, self)
        self.transforms = StaticTransformBroadcaster(self)
        self.map_transforms = {}
        self.group = ReentrantCallbackGroup()
        self.spawn = self.create_client(SpawnEntity, f'/world/{self.world}/create', callback_group=self.group)
        self.set_pose = self.create_client(SetEntityPose, f'/world/{self.world}/set_pose', callback_group=self.group)
        self.power = self.create_publisher(BatteryState, 'mainboard/power_sensor/power', 10)
        self.board = self.create_publisher(Bool, 'propulsion_board/status', 10)
        self.enable = self.create_publisher(Bool, 'enable', 10)
        self.angle = self.create_publisher(Float32, 'acoustic/hydrophone/base_angle', 10)
        self.spawn_events = self.create_publisher(String, 'sim/spawn_events', 10)
        self.gripper = self.create_publisher(UInt16, 'sim/gripper_angle', 10)
        self.create_service(SetBool, 'sim/set_enabled', self.set_enabled)
        self.create_service(SetObjectTransform, 'map/set_object_transform', self.set_transform)
        self.create_subscription(Odometry, f'/model/{self.get_namespace().strip("/")}/odometry', self.on_truth, 10)
        self.create_subscription(Float32, 'actuators/ball_dropper/set_angle', self.drop, 10)
        for topic in ('actuators/gripper/set_angle', 'actuators/gripper1/set_angle'):
            self.create_subscription(UInt16, topic, self.gripper.publish, 10)
        for index in (1, 2):
            self.create_service(Trigger, f'actuators/torpedo_{index}/launch',
                                partial(self.torpedo, index), callback_group=self.group)
        for item in ('bandaid', 'electric', 'nutbolt', 'pill'):
            self.create_service(Trigger, f'actuators/{item}/spawn',
                                partial(self.bottle, item), callback_group=self.group)
        self.create_timer(0.1, self.publish_status)
        self.get_logger().info('Ideal mocks: 16 V, healthy board, enable=false; sim/set_enabled controls enable. '
                               'Gripper is command echo; torpedoes use ideal 1 m/s pose motion, without propulsion/collision dynamics.')

    def set_enabled(self, request, response):
        self.enabled = request.data
        response.success = True
        response.message = f'simulation enable={self.enabled}'
        return response

    def set_transform(self, request, response):
        transform = request.transform
        if not transform.child_frame_id or not transform.header.frame_id:
            response.success = False
            response.message = 'parent and child frames are required'
            return response
        self.map_transforms[transform.child_frame_id] = transform
        self.transforms.sendTransform(list(self.map_transforms.values()))
        response.success = True
        response.message = 'simulation static map transform stored'
        return response

    def on_truth(self, message):
        self.truth = message.pose.pose

    def publish_status(self):
        self.power.publish(BatteryState(voltage=16.0))
        self.board.publish(Bool(data=True))
        self.enable.publish(Bool(data=self.enabled))
        if self.truth:
            p, q = self.truth.position, self.truth.orientation
            dx, dy, dz = self.target[0]-p.x, self.target[1]-p.y, -1.95-p.z
            local_x = (1-2*(q.y*q.y+q.z*q.z))*dx + 2*(q.x*q.y+q.w*q.z)*dy + 2*(q.x*q.z-q.w*q.y)*dz
            local_y = 2*(q.x*q.y-q.w*q.z)*dx + (1-2*(q.x*q.x+q.z*q.z))*dy + 2*(q.y*q.z+q.w*q.x)*dz
            angle = math.atan2(local_y, local_x)
            self.angle.publish(Float32(data=math.atan2(math.sin(angle), math.cos(angle))))

        now = self.get_clock().now().nanoseconds * 1e-9
        for name, (initial, born) in list(self.projectiles.items()):
            age = now-born
            if age > 20.0 or age < 0:
                self.projectiles.pop(name, None)
                continue
            if name in self.pose_pending and not self.pose_pending[name].done():
                continue
            request = SetEntityPose.Request()
            request.entity.name = name
            request.entity.type = request.entity.MODEL
            request.pose = copy.deepcopy(initial)
            q = initial.orientation
            request.pose.position.x += age*(1-2*(q.y*q.y+q.z*q.z))
            request.pose.position.y += age*2*(q.x*q.y+q.w*q.z)
            request.pose.position.z += age*2*(q.x*q.z-q.w*q.y)
            if self.set_pose.service_is_ready():
                self.pose_pending[name] = self.set_pose.call_async(request)

    def frame_pose(self, frame):
        transform = self.buffer.lookup_transform('odom', frame, Time())
        pose = Pose()
        pose.position.x = transform.transform.translation.x
        pose.position.y = transform.transform.translation.y
        pose.position.z = transform.transform.translation.z
        pose.orientation = transform.transform.rotation
        return pose

    async def spawn_model(self, relative_path, pose, key):
        if not self.spawn.service_is_ready():
            return False, 'Gazebo create service is not ready'
        root = ET.parse(Path(get_package_share_directory('auv_sim_description')) / 'models' / relative_path).getroot()
        # Resolve package-local mesh URIs via GZ_SIM_RESOURCE_PATH. Keep source model untouched.
        if key.startswith('torpedo'):
            model = root.find('model')
            static = model.find('static')
            if static is None:
                static = ET.SubElement(model, 'static')
            static.text = 'true'
        self.serial += 1
        request = SpawnEntity.Request()
        request.entity_factory.name = f'mock_{key}_{self.serial}'
        request.entity_factory.sdf = ET.tostring(root, encoding='unicode')
        request.entity_factory.pose = pose
        future = self.spawn.call_async(request)
        # rclpy futures are awaitable within executor tasks.
        timer = self.create_timer(5.0, lambda: future.cancel() if not future.done() else None,
                                  clock=Clock(clock_type=ClockType.STEADY_TIME), callback_group=self.group)
        try:
            response = await future
            if response is None:
                return False, 'Gazebo create timed out; creation state unknown'
            self.spawn_events.publish(String(data=json.dumps({'key': key, 'name': request.entity_factory.name, 'success': response.success})))
            return response.success, request.entity_factory.name
        except CancelledError:
            return False, 'Gazebo create timed out; creation state unknown'
        finally:
            self.destroy_timer(timer)

    async def torpedo(self, index, request, response):
        namespace = self.get_namespace().strip('/')
        if namespace == 'taluy_mini':
            response.success = False
            response.message = 'Mini has no torpedo launcher frames'
            return response
        frame = 'torpedo_upper_link' if index == 1 else 'torpedo_bottom_link'
        try:
            pose = self.frame_pose(f'{namespace}/base_link/{frame}')
            response.success, response.message = await self.spawn_model('robosub_torpedo/torpedo.sdf', pose, f'torpedo{index}')
            if response.success:
                self.projectiles[response.message] = (copy.deepcopy(pose), self.get_clock().now().nanoseconds * 1e-9)
        except TransformException as error:
            response.success, response.message = False, str(error)
        return response

    async def bottle(self, item, request, response):
        pose = Pose()
        pose.orientation.w = 1.0
        pose.position.x, pose.position.y, pose.position.z = 13.0, -4.25, -1.3
        response.success, response.message = await self.spawn_model(f'{item}/model.sdf', pose, item)
        return response

    def drop(self, message):
        if abs(message.data) <= 1e-3:
            return
        index = 1 if message.data > 0 else 2
        if index in self.dropped or index in self.inflight:
            return
        try:
            pose = self.frame_pose(f'{self.get_namespace().strip("/")}/base_link/ball_dropper_{index}_link')
        except TransformException as error:
            self.get_logger().warning(str(error))
            return
        self.inflight.add(index)
        async def perform():
            try:
                ok, detail = await self.spawn_model('robosub_bin/ball.sdf', pose, f'ball{index}')
                if ok:
                    self.dropped.add(index)
                self.get_logger().info(f'ball drop success={ok}: {detail}')
            finally:
                self.inflight.discard(index)
        self.executor.create_task(perform)


def main():
    rclpy.init()
    node = SimulationMocks()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

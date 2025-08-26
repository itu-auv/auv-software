#!/usr/bin/env python3
import rospy, tf2_ros
from geometry_msgs.msg import TransformStamped
from std_srvs.srv import Trigger, TriggerResponse
import yaml
import numpy as np
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf2_ros import StaticTransformBroadcaster


class PipeFramePlanner:
    def __init__(self):
        self.cfg_path = rospy.get_param(
            "~rules_yaml", rospy.get_param("~rules", "config/pipe_rules.yaml")
        )
        raw = yaml.safe_load(open(self.cfg_path, "r"))
        self.cfg = raw.get("pipe_frame_planner", raw)

        self.odom = self.cfg["tf"]["odom"]
        self.base_link = self.cfg["tf"]["base_link"]
        self.frames = self.cfg["frames"]
        self.rules = self.cfg["path_rules"]

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(60.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        self.stb = StaticTransformBroadcaster()

        self.start_created = False
        self.z_start = None

        rospy.Service("/pipe_plan/create_start", Trigger, self.srv_create_start)
        rospy.Service("/pipe_plan/commit_bootstrap", Trigger, self.srv_commit_bootstrap)
        rospy.Service("/pipe_plan/align_and_build", Trigger, self.srv_align_and_build)

    def make_static_tf(self, parent, child, xyz, rpy):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = xyz[0]
        t.transform.translation.y = xyz[1]
        t.transform.translation.z = xyz[2]
        q = quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        (
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ) = q
        self.stb.sendTransform(t)

    def srv_create_start(self, req):
        try:
            tfbl = self.tfbuf.lookup_transform(
                self.odom, self.base_link, rospy.Time(0), rospy.Duration(2.0)
            )
            x = tfbl.transform.translation.x
            y = tfbl.transform.translation.y
            z = tfbl.transform.translation.z
            q = tfbl.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            self.z_start = z + float(self.cfg["plan"]["default_z_offset_m"])
            self.make_static_tf(
                self.odom,
                self.frames["start_frame_name"],
                (x, y, self.z_start),
                (0, 0, yaw),
            )
            self.start_created = True
            return TriggerResponse(success=True, message="start created")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def srv_commit_bootstrap(self, req):
        if not self.start_created or self.z_start is None:
            return TriggerResponse(success=False, message="start not created yet")
        try:
            tfbl = self.tfbuf.lookup_transform(
                self.odom, self.base_link, rospy.Time(0), rospy.Duration(2.0)
            )
            x = tfbl.transform.translation.x
            y = tfbl.transform.translation.y
            q = tfbl.transform.rotation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

            self.make_static_tf(
                self.odom,
                self.frames["bootstrap_frame_name"],
                (x, y, self.z_start),
                (0, 0, yaw),
            )
            return TriggerResponse(success=True, message="bootstrap created")
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))

    def srv_align_and_build(self, req):
        try:
            # get start & bootstrap in odom
            s = self.tfbuf.lookup_transform(
                self.odom,
                self.frames["start_frame_name"],
                rospy.Time(0),
                rospy.Duration(2.0),
            )
            b = self.tfbuf.lookup_transform(
                self.odom,
                self.frames["bootstrap_frame_name"],
                rospy.Time(0),
                rospy.Duration(2.0),
            )
            sx, sy = s.transform.translation.x, s.transform.translation.y
            bx, by = b.transform.translation.x, b.transform.translation.y
            yaw = np.arctan2(by - sy, bx - sx)  # boru yönü

            # aligned start at same origin (sx,sy,z_start) but corrected yaw
            self.make_static_tf(
                self.odom,
                self.frames["aligned_start_name"],
                (sx, sy, self.z_start),
                (0, 0, yaw),
            )

            wp_prefix = self.frames["waypoint_prefix"]
            idx = 1
            cur_yaw = yaw
            cur_x, cur_y, cur_z = sx, sy, self.z_start

            for rule in self.rules:
                if rule["type"] == "turn":
                    # update heading only
                    cur_yaw += np.deg2rad(float(rule["deg"]))
                elif rule["type"] == "forward":
                    dist = float(rule["meters"])
                    # move along current heading
                    cur_x += dist * np.cos(cur_yaw)
                    cur_y += dist * np.sin(cur_yaw)

                    name = f"{wp_prefix}{idx:02d}"
                    self.make_static_tf(
                        self.odom, name, (cur_x, cur_y, cur_z), (0, 0, cur_yaw)
                    )
                    idx += 1
                else:
                    rospy.logwarn("Unknown rule: %s", rule)

            return TriggerResponse(
                success=True, message=f"aligned & waypoints built ({idx-1})"
            )
        except Exception as e:
            return TriggerResponse(success=False, message=str(e))


if __name__ == "__main__":
    rospy.init_node("pipe_frame_publisher_node")
    PipeFramePlanner()
    rospy.spin()

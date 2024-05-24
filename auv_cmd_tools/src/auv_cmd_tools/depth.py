import rospy
from std_msgs.msg import Float32


class DepthClient:
    def __init__(self, depth_topic="/turquoise/sensors/pressure/depth", target_depth_topic="/turquoise/cmd_depth", timeout=1.5):
        self.depth_topic = depth_topic
        self.target_depth_topic = target_depth_topic
        self.timeout = timeout

    def get_depth(self):
        depth = rospy.wait_for_message(self.depth_topic, Float32, self.timeout).data
        return depth

    def set_depth(self, depth):
        rate = rospy.Rate(20.0)
        pub = rospy.Publisher(self.target_depth_topic, Float32, queue_size=1)

        while not rospy.is_shutdown():
            msg = Float32(depth)
            pub.publish(msg)
            rate.sleep()


class SetDepthClientParser:
    def __init__(self, parser=None, topic="/turquoise/set_arming"):
        self.topic = topic
        # parser = argparse.ArgumentParser(usage="""some usage""")
        parser.add_argument("depth", type=float,
                            help="target depth in meters.")
        self.args = parser.parse_args()

        print("Setting depth to", self.args.depth)

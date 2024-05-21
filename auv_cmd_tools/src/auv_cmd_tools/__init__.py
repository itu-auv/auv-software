#!/usr/bin/env python3
from __future__ import print_function
import argparse
from .arm import ArmingClient
import rospy
import logging
from .depth import DepthClient
import argcomplete
import subprocess
import os
from std_srvs.srv import Trigger, TriggerRequest
from auv_msgs.srv import FirmwareUpdate, FirmwareUpdateRequest
from auv_msgs.msg import Power
from hashlib import md5
import shutil


bringup_pkg = os.getenv("AUV_BRINGUP_PKG")
bringup_features = bringup_pkg is not None 
if bringup_pkg is None:
    print("Environment variable AUV_BRINGUP_PKG is not set. Bringup features are disabled")
class Parameters:
    ARM_SERVICE_TOPIC = "/turquoise/set_arming"
    ARMED_MSG_TOPIC = "/turquoise/is_armed"
    DEPTH_TOPIC = "/turquoise/sensors/pressure/depth"
    TARGET_DEPTH_TOPIC = "/turquoise/cmd_depth"
    TIMEOUT = 2.0


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class AuvCmdTool:
    def __init__(self):
        pass

    def __setup__(self, args):
        self.arm_client = ArmingClient(Parameters.ARM_SERVICE_TOPIC,
                                       Parameters.ARMED_MSG_TOPIC, timeout=args.timeout)
        self.depth_client = DepthClient(depth_topic=Parameters.DEPTH_TOPIC,
                                        target_depth_topic=Parameters.TARGET_DEPTH_TOPIC, timeout=args.timeout)

    def get_armed(self, args):

        res = self.arm_client.is_armed()
        if res is None:
            rospy.logerr("Unable to get armed information")
        else:
            print("Vehicle Armed Status:{}".format(res))

    def get_depth(self, args):
        try:
            depth = self.depth_client.get_depth()
            print("Current depth:", depth)
        except Exception as e:
            rospy.logerr("Timeout reached while waiting for message. Error: {}".format(e))

    def arm(self, args):
        args.armed = "true"
        return self.set_armed(args)

    def disarm(self, args):
        args.armed = "false"
        self.set_armed(args)

    def set_armed(self, args):
        if self.arm_client.set_arm(args.armed != "false"):
            print("Setting arming to:", args.armed, "was success.")
        else:
            rospy.logerr(
                "Setting arming to: {}, has failed".format(str2bool(args.armed)))

    def set_depth(self, args):
        print("Setting depth to", args.depth)
        self.depth_client.set_depth(args.depth)

    def sync_bags(self, args):
        cmd = """rsync -a --ignore-existing --progress {}:{} {} -vv""".format(args.remote, args.remote_path, args.local_path)
        p = subprocess.Popen(cmd.split(" "))
        p.communicate()

    def set_speed(self, args):
        print("Setting speed to", args.speed)

    def test_thrusters(self, args):
        os.system("rosrun auv_mainboard_bridge motor_tester.py")

    def get_voltage(self, args):
        msg = rospy.wait_for_message("/turquoise/battery/power", Power)
        print("Voltage: {0:.2f}V".format(msg.voltage))

    def get_current(self, args):
        msg = rospy.wait_for_message("/turquoise/battery/power", Power)
        print("Current: {0:.2f}A".format(msg.current))
    
    def get_power(self, args):
        msg = rospy.wait_for_message("/turquoise/battery/power", Power)
        print("Power: {0:.2f}W".format(abs(msg.power)))

    def check_io(self, args):
        os.system("rosrun {} check_io".format(bringup_pkg))

    def init(self, args):
        os.system("roslaunch {} start_nocams.launch".format(bringup_pkg))

    def cams(self, args):
        os.system("roslaunch {} cameras.launch".format(bringup_pkg))

    def jetson(self, args):
        if args.jetson_cmd == "set-profile":
            profile = args.profile
            os.system("roslaunch auv_bringup initialize.launch profile:={}".format(profile))

    def control(self, args):
        if args.control_cmd == "enable":
            os.system('rosservice call /turquoise/control/set_enabled "data: 1"')
        elif args.control_cmd == "disable":
            os.system('rosservice call /turquoise/control/set_enabled "data: 0"')

    def dvl(self, args):
        if args.dvl_cmd == "enable":
            os.system('rosservice call /turquoise/sensors/dvl/set_ping "data: 1"')
        elif args.dvl_cmd == "disable":
            os.system('rosservice call /turquoise/sensors/dvl/set_ping "data: 0"')

    def board(self, args):
        if args.board_cmd == "fw-update":
            fname = args.file
            def let_user_pick(options):
                print("Please choose firmware file:")
                for idx, element in enumerate(options):
                    print("{}) {}".format(idx+1,element))
                i = input("Enter number: ")
                try:
                    if 0 < int(i) <= len(options):
                        return options[int(i) - 1]
                except:
                    pass
                return None
            def upload(filaneme):
                print("Uploading '{}' to board mcu.".format(filaneme))
                srv = rospy.ServiceProxy("/turquoise/board/fw_update", FirmwareUpdate)
                file = open(filaneme, "rb")
                data = file.read()
                file.close()
                req = FirmwareUpdateRequest()
                req.data = data
                req.size = len(data)
                req.md5sum = md5(data).hexdigest()
                resp = srv.call(req)
                print(resp.success)
            if fname == "latest":
                down_path = "/tmp/fw_build/"
                down_file = os.path.join(down_path, "build.zip")
                if os.path.exists(down_path):
                    shutil.rmtree(down_path)
                os.system("gh release -R itu-auv/mainboard-firmware download -p 'build.zip' -D {}".format(down_path))
                os.system("unzip {} -d {}".format(down_file, down_path))
                bin_path = os.path.join(down_path, "build")
                files = [x for x in os.listdir(bin_path) if os.path.isfile(os.path.join(bin_path, x))]
                selection = let_user_pick(files)
                if selection is not None:
                    fw_file = os.path.join(bin_path, selection)
                    upload(fw_file)
                shutil.rmtree(down_path)
            else:
                upload(fname)
        elif args.board_cmd == "reset":
            srv = rospy.ServiceProxy("/turquoise/board/reset", Trigger)
            resp = srv.call(TriggerRequest())
            print(resp.success)
        else:
            print("Unknown command: {}".format(args.board_cmd))

    def main(self):
        parser = argparse.ArgumentParser(usage="""auv <command> [<args>]""")

        parser.add_argument("-v", "--verbose",
                            action="store_true", help="Enable verbose output")

        parser.add_argument("-t", "--timeout", type=float,
                            default=1.5, help="sets Message/Service connection timeout for <TIMEOUT> seconds.")

        subparsers = parser.add_subparsers(
            dest="command", help="sub-command help")

        subparsers.add_parser("arm", help="Arms vehicle")
        subparsers.add_parser("disarm", help="Disarms vehicle")
        subparsers.add_parser("test_thrusters", help="Tests thrusters and directions")
        subparsers.add_parser("get_voltage", help="Get battery voltage")
        subparsers.add_parser("get_current", help="Get battery current")
        subparsers.add_parser("get_power", help="Get consumed power")
        subparsers.add_parser("check_io", help="Check all I/O")
        if bringup_features:
            subparsers.add_parser("init", help="Start bare-bone packages")
            subparsers.add_parser("cams", help="Start camera packages")

        set_depth_parser = subparsers.add_parser(
            "set_depth", help="set_depth command help")
        set_depth_parser.add_argument(
            "depth", type=float, help="target depth in meters")

        set_armed_parser = subparsers.add_parser(
            "set_armed", help="set_armed command help")
        set_armed_parser.add_argument(
            "armed", type=str2bool, nargs="?", default=False, const=True, help="target armed value")

        control_parser = subparsers.add_parser(
            "control", help="control commands"
        )
        control_sbp = control_parser.add_subparsers(dest="control_cmd", help="control commands")
        control_sbp.add_parser("enable", help="Enable controllers")
        control_sbp.add_parser("disable", help="Disable controllers")

        dvl_parser = subparsers.add_parser(
            "dvl", help="dvl commands"
        )
        dvl_sbp = dvl_parser.add_subparsers(dest="dvl_cmd", help="dvl commands")
        dvl_sbp.add_parser("enable", help="start ping")
        dvl_sbp.add_parser("disable", help="stop ping")

        board_parser = subparsers.add_parser(
            "board", help="board commands")
        sbp = board_parser.add_subparsers(dest="board_cmd", help="board commands")
        fw_parser = sbp.add_parser("fw-update", help="perform firmware update")
        fw_parser.add_argument("file", type=str, help="path to .bin file")
        sbp.add_parser("reset", help="reset MCU")

        # set_speed_parser = subparsers.add_parser("set_speed", help="set command help")
        # set_speed_parser.add_argument("speed", type=float, help="target speed in m/s")

        subparsers.add_parser(
            "get_armed", help="returns current arming status")

        subparsers.add_parser("get_depth", help="returns current depth status")

        sync_bags_parser = subparsers.add_parser("sync_bags", help="Transfer .bag files from vehicle computer")
        sync_bags_parser.add_argument("-r", "--remote", type=str, help="remote url, ex: nvidia@jetson.local", default="nvidia@jetson.local")
        bagdir = "/home/nvidia/bags"
        sync_bags_parser.add_argument("-p", "--remote-path", type=str, help="remote path, ex: ~/bags", default=bagdir)
        sync_bags_parser.add_argument("-l", "--local-path", type=str, help="local path, ex: .", default=".")
        # # subsubparsers = set_parser.add_subparsers(dest="command2", help="sub-command help")
        # # set_parser = subsubparsers.add_parser("depth", help="set command help")
        # set_parser.add_argument("command2", type=float, help="target depth value in meters.")

        # set_depth_parser = subparsers.add_parser("set_depth", help="set_depth command help")
        # set_depth_parser.add_argument("depth", type=float, help="target depth value in meters.")

        # set_speed_parser = subparsers.add_parser("set_speed", help="set_speed command help")
        # set_speed_parser.add_argument("speed", type=float, help="target speed in m/s.")


        jetson_parser = subparsers.add_parser(
            "jetson", help="jetson commands")
        jp = jetson_parser.add_subparsers(dest="jetson_cmd", help="jetson commands")
        fw_parser = jp.add_parser("set-profile", help="set jetson clock profile")
        fw_parser.add_argument("profile", type=str, help="profile name: silent, default, max")



        argcomplete.autocomplete(parser)
        args = parser.parse_args()
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        import time
        rospy.init_node("auv_tools_node_{}".format(time.time()).replace(".", "_"))
        if args.verbose:
            rospy.logwarn("Verbosity set to: {}".format(args.verbose))
        self.__setup__(args)
        getattr(self, args.command)(args)

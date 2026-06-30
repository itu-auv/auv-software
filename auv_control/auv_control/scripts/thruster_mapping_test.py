#!/usr/bin/env python3
import argparse
import sys
from typing import List

import rospy
from std_msgs.msg import UInt16MultiArray


def parse_mapping(value: str) -> List[int]:
    try:
        items = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "mapping must be comma-separated integers, e.g. 0,1,4,5,7,3,6,2"
        ) from exc
    return items


def load_mapping(namespace: str, override: List[int]) -> List[int]:
    if override:
        return override

    ns = namespace.strip("/")
    candidates = [
        f"/{ns}/thruster_manager_node/mapping",
        f"/{ns}/mapping",
        "~mapping",
    ]
    for param_name in candidates:
        if rospy.has_param(param_name):
            return [int(item) for item in rospy.get_param(param_name)]

    raise RuntimeError(
        "Could not find mapping on the ROS param server. "
        "Start auv_control first, or pass --mapping 0,1,4,5,7,3,6,2."
    )


def validate_mapping(mapping: List[int]) -> None:
    expected = list(range(8))
    if sorted(mapping) != expected:
        raise ValueError(f"mapping must be a permutation of {expected}; got {mapping}")


def publish_for_duration(pub, data: List[int], duration: float, rate_hz: float) -> None:
    rate = rospy.Rate(rate_hz)
    end_time = rospy.Time.now() + rospy.Duration.from_sec(duration)
    msg = UInt16MultiArray(data=data)
    while not rospy.is_shutdown() and rospy.Time.now() < end_time:
        pub.publish(msg)
        rate.sleep()


def run_single_test(
    args, pub, mapping: List[int], logical_motor_zero_based: int
) -> None:
    drive_slot = mapping.index(logical_motor_zero_based)
    neutral = [args.neutral] * 8
    command = neutral.copy()
    command[drive_slot] = args.pwm

    rospy.loginfo(
        "Logical motor %d -> drive_pulse slot %d because mapping[%d] = %d",
        logical_motor_zero_based + 1,
        drive_slot + 1,
        drive_slot,
        logical_motor_zero_based,
    )
    rospy.loginfo("Publishing drive_pulse data: %s", command)

    if args.dry_run:
        return

    publish_for_duration(pub, neutral, args.neutral_duration, args.rate)
    publish_for_duration(pub, command, args.duration, args.rate)
    publish_for_duration(pub, neutral, args.neutral_duration, args.rate)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test thruster mapping by commanding a logical motor number and "
            "publishing the mapped /board/drive_pulse slot."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--motor",
        type=int,
        help="Logical motor number to test, using 1-based numbering from the robot labels.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Test logical motors 1 through 8 in order.",
    )
    parser.add_argument("--namespace", default="taluy", help="Robot ROS namespace.")
    parser.add_argument(
        "--mapping",
        type=parse_mapping,
        help="Override mapping as comma-separated zero-based values.",
    )
    parser.add_argument(
        "--pwm", type=int, default=1600, help="PWM sent to the tested motor."
    )
    parser.add_argument("--neutral", type=int, default=1500, help="Neutral PWM value.")
    parser.add_argument(
        "--duration", type=float, default=0.75, help="Seconds to publish the test PWM."
    )
    parser.add_argument(
        "--neutral-duration",
        type=float,
        default=0.35,
        help="Seconds to publish neutral before and after each test.",
    )
    parser.add_argument(
        "--pause", type=float, default=1.5, help="Pause between --all tests."
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Publish rate in Hz.")
    parser.add_argument(
        "--topic",
        help="Drive pulse topic. Defaults to /<namespace>/board/drive_pulse.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print only; do not publish."
    )
    args = parser.parse_args(argv)

    if args.motor is not None and not 1 <= args.motor <= 8:
        parser.error("--motor must be between 1 and 8")
    if not 1100 <= args.pwm <= 1900:
        parser.error("--pwm must be between 1100 and 1900")
    if not 1100 <= args.neutral <= 1900:
        parser.error("--neutral must be between 1100 and 1900")
    if args.duration <= 0.0 or args.neutral_duration < 0.0 or args.rate <= 0.0:
        parser.error("durations and rate must be positive")
    return args


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    rospy.init_node("thruster_mapping_test", anonymous=True)

    mapping = load_mapping(args.namespace, args.mapping)
    validate_mapping(mapping)
    rospy.loginfo("Using mapping slot->logical: %s", mapping)

    topic = args.topic or f"/{args.namespace.strip('/')}/board/drive_pulse"
    pub = rospy.Publisher(topic, UInt16MultiArray, queue_size=1)
    rospy.sleep(0.5)

    if args.all:
        for logical_motor_zero_based in range(8):
            run_single_test(args, pub, mapping, logical_motor_zero_based)
            if logical_motor_zero_based != 7 and not args.dry_run:
                rospy.sleep(args.pause)
    else:
        run_single_test(args, pub, mapping, args.motor - 1)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:
        rospy.logerr("thruster mapping test failed: %s", exc)
        sys.exit(1)

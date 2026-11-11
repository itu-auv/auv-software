#!/usr/bin/env python3

import rclpy
from auv_sim_vision.bbox import SimBboxNode


def main():
    node = SimBboxNode()
    node.run()


if __name__ == "__main__":
    main()

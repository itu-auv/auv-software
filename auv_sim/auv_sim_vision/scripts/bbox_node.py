#!/usr/bin/env python3

import rospy
from auv_sim_vision.bbox import SimBboxNode


if __name__ == "__main__":
    try:
        node = SimBboxNode()
        node.run()
    except rospy.ROSInterruptException:
        pass

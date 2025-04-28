#!/usr/bin/env python3
import rospy
from octomap_msgs.srv import GetOctomap
import octomap


def main():
    rospy.init_node("octomap_z_stats")
    rospy.loginfo("Waiting for /octomap_binary service...")
    rospy.wait_for_service("/octomap_binary")

    try:
        get_map = rospy.ServiceProxy("/octomap_binary", GetOctomap)
        resp = get_map()
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)
        return

    # Build a local OcTree at the published resolution
    tree = octomap.OcTree(resp.map.resolution)
    tree.readBinary(resp.map.data)

    # Query the metric bounding box
    min_bb = tree.getMetricMin()
    max_bb = tree.getMetricMax()

    rospy.loginfo(
        "OctoMap Z‐bounds → min_z = %.3f m,   max_z = %.3f m", min_bb.z(), max_bb.z()
    )


if __name__ == "__main__":
    main()

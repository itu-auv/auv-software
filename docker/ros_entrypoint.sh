#!/bin/bash
set -e

# Source ROS setup (works for both bash and zsh downstream)
source /opt/ros/noetic/setup.bash
[ -f /auv_ws/devel/setup.bash ] && source /auv_ws/devel/setup.bash

exec "$@"

# Desktop dev image for the ROS 2 port. Switch distro with --build-arg ROS_DISTRO=jazzy.
ARG ROS_DISTRO=lyrical
FROM docker.io/osrf/ros:${ROS_DISTRO}-desktop-full

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git curl vim less gdb clang-format \
    python3-pip python3-vcstool python3-colcon-common-extensions python3-rosdep \
    python3-numpy python3-yaml python3-scipy python3-opencv python3-transforms3d \
    python3-pyqt5 python3-tk python3-zmq python3-pandas python3-matplotlib python3-pyflakes \
    libeigen3-dev libfmt-dev libpcl-dev \
    ros-${ROS_DISTRO}-ros-gz \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-smach ros-${ROS_DISTRO}-smach-ros \
    ros-${ROS_DISTRO}-tf-transformations \
    ros-${ROS_DISTRO}-marine-acoustic-msgs \
    ros-${ROS_DISTRO}-vision-msgs \
    ros-${ROS_DISTRO}-image-pipeline ros-${ROS_DISTRO}-image-transport-plugins \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-pcl-ros ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-angles ros-${ROS_DISTRO}-xacro \
    ros-${ROS_DISTRO}-joint-state-publisher ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-robot-state-publisher \
    ros-${ROS_DISTRO}-joy ros-${ROS_DISTRO}-rqt-reconfigure ros-${ROS_DISTRO}-rqt-image-view \
    ros-${ROS_DISTRO}-realsense2-camera ros-${ROS_DISTRO}-rosbag2 \
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# Generated Python messages with a field named `range` call that field instead of the
# built-in (marine_acoustic_msgs/Dvl). Remove once https://github.com/ros2/rosidl_python/pull/276 is released.
RUN grep -rl --include='_*.py' 'for x in range(' /opt/ros/${ROS_DISTRO}/lib/python3*/site-packages \
    | xargs -r sed -i 's/for x in range(/for x in builtins.range(/'

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /etc/bash.bashrc && \
    echo '[ -f /auv_ws/install/setup.bash ] && source /auv_ws/install/setup.bash' >> /etc/bash.bashrc

WORKDIR /auv_ws

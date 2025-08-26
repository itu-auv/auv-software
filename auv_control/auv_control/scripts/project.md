Hello, new project, new node. Determine the name of the node at the end by the concept of it:

## the concept

The node will have the parameters: starting_coordinate_gps, target_coordinate_gps, starting_coordinate_cartesian, target_coordinate_cartesian

these parameters are input. make them dynamically configurable (the cfg method). One example of how to do that is at @slalom_trajectory_publisher.py

the node's goal is to create a target frame at the location of target coordinate.

The node will also will now in which way is north. For now, assume north is hardcoded to somewhere. How to get that info will be implemented later

if gps is given, do these:

start coordinates are where the robot is at the moment of start (when the service for the node is called) (localization gives the odom - taluy transform).

1. turn the gps info to a target frame in odom frame.
2. broadcast the frame with the same method as slalom trajectory publisher.


dont mind the cartesian coordinates for now.

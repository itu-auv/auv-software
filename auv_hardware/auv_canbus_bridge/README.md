# auv_canbus_bridge


- [auv\_canbus\_bridge](#auv_canbus_bridge)
  - [setup virtual can for testing](#setup-virtual-can-for-testing)


## setup virtual can for testing
sudo modprobe vcan
sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0
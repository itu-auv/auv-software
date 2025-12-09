#!/bin/bash

# Test script to manually verify teleop setup

echo "=== Teleop Diagnostic Script ==="
echo ""

# Check if joystick device exists
echo "1. Checking for joystick devices..."
if ls /dev/input/js* > /dev/null 2>&1; then
    ls -la /dev/input/js*
    echo "✓ Joystick device(s) found"
else
    echo "✗ No joystick devices found!"
    echo "  Make sure your Xbox controller is plugged in"
    exit 1
fi

echo ""
echo "2. Testing joystick with jstest (press Ctrl+C to stop)..."
echo "   Move sticks and press buttons - you should see values changing"
echo ""
read -p "Press Enter to start jstest on /dev/input/js1 (or Ctrl+C to skip)..."
jstest /dev/input/js1

echo ""
echo "3. To manually start teleop from command line:"
echo "   roslaunch auv_teleop start_teleop.launch namespace:=taluy controller:=xbox id:=1"
echo ""
echo "4. To check if teleop is publishing:"
echo "   rostopic echo /taluy/cmd_vel"
echo "   rostopic echo /taluy/enable"
echo ""
echo "5. Common issues:"
echo "   - Permission denied: sudo chmod a+rw /dev/input/js1"
echo "   - Wrong device: check which js device your controller is (js0, js1, etc.)"
echo "   - Controller not detected: check dmesg | grep -i xbox"

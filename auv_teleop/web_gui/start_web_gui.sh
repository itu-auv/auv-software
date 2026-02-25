#!/bin/bash

# ITU AUV Web GUI Startup Script
# This script launches all necessary services for the web control panel

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WEB_GUI_DIR="${SCRIPT_DIR}"

echo "========================================="
echo "🚢 ITU AUV Web GUI Startup"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if roscore is running
echo -n "Checking ROS master... "
if ! rostopic list &> /dev/null; then
    echo -e "${YELLOW}⚠${NC}"
    echo ""
    echo -e "${YELLOW}Warning: ROS master is not running yet${NC}"
    echo "The web GUI will start, but won't connect to ROS until you:"
    echo ""
    echo "  1. Start simulation or roscore:"
    echo "     roslaunch auv_sim_bringup start_gazebo.launch"
    echo ""
    echo "  2. Refresh the web page to reconnect"
    echo ""
    ROSCORE_RUNNING=false
else
    echo -e "${GREEN}✓${NC}"
    ROSCORE_RUNNING=true
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check dependencies
echo ""
echo "Checking dependencies..."

if ! command_exists npm; then
    echo -e "${RED}✗ npm not found${NC}"
    echo "Please install Node.js and npm first"
    exit 1
fi
echo -e "${GREEN}✓${NC} npm found"

if ! command_exists roslaunch; then
    echo -e "${RED}✗ ROS not found${NC}"
    echo "Please source your ROS workspace"
    exit 1
fi
echo -e "${GREEN}✓${NC} ROS found"

# Check if web_video_server package is installed
if ! rospack find web_video_server &> /dev/null; then
    echo -e "${YELLOW}⚠ web_video_server not found${NC}"
    echo "Camera streaming will not be available."
    echo "Install with: sudo apt-get install ros-noetic-web-video-server"
    WEB_VIDEO_AVAILABLE=false
else
    echo -e "${GREEN}✓${NC} web_video_server found"
    WEB_VIDEO_AVAILABLE=true
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "========================================="
    echo "Shutting down services..."
    echo "========================================="
    
    if [ ! -z "$ROSLAUNCH_PID" ]; then
        echo "Stopping rosbridge & web_video_server..."
        kill $ROSLAUNCH_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$NPM_PID" ]; then
        echo "Stopping Vite dev server..."
        kill $NPM_PID 2>/dev/null || true
    fi
    
    echo "Cleanup complete"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# Start rosbridge and web_video_server
echo ""
echo "========================================="
echo "Starting ROS services..."
echo "========================================="

if [ "$ROSCORE_RUNNING" = true ]; then
    # roscore is running, start immediately
    echo -e "${GREEN}✓${NC} Starting ROS services..."
    echo ""
    roslaunch auv_teleop web_gui.launch 2>&1 | sed 's/^/[ROS] /' &
    ROSLAUNCH_PID=$!
    
    if [ "$WEB_VIDEO_AVAILABLE" = true ]; then
        echo -e "${GREEN}✓${NC} rosbridge_server and web_video_server starting"
    else
        echo -e "${GREEN}✓${NC} rosbridge_server starting"
    fi

    # Wait for rosbridge to be ready
    echo ""
    echo -n "Waiting for rosbridge to be ready"
    for i in {1..15}; do
        if nc -z localhost 9090 2>/dev/null; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep 1
    done
    echo ""
else
    # roscore not running yet, start a background monitor to launch when ready
    echo -e "${YELLOW}⚠${NC} Waiting for roscore to start ROS services..."
    echo "  Starting background monitor (will connect automatically)"
    echo ""
    
    # Background process to wait for roscore and then launch
    (
        while ! rostopic list &> /dev/null; do
            sleep 2
        done
        echo ""
        echo -e "${GREEN}✓${NC} roscore detected! Starting ROS services..."
        echo ""
        roslaunch auv_teleop web_gui.launch 2>&1 | sed 's/^/[ROS] /'
    ) &
    ROSLAUNCH_PID=$!
fi

# Start Vite dev server
echo ""
echo "========================================="
echo "Starting Vite dev server..."
echo "========================================="
cd "${WEB_GUI_DIR}"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

npm run dev &
NPM_PID=$!

# Wait a bit for Vite to start
sleep 3

echo ""
echo "========================================="
if [ "$ROSCORE_RUNNING" = true ]; then
    echo -e "${GREEN}🎉 All services started successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Web GUI started (waiting for ROS)${NC}"
fi
echo "========================================="
echo ""
echo "Services running:"
if [ "$ROSCORE_RUNNING" = true ]; then
    echo "  • ROS Bridge:       ws://localhost:9090"
    if [ "$WEB_VIDEO_AVAILABLE" = true ]; then
        echo "  • Video Server:     http://localhost:8080"
    fi
else
    echo -e "  • ROS Bridge:       ${YELLOW}Waiting for roscore...${NC}"
    if [ "$WEB_VIDEO_AVAILABLE" = true ]; then
        echo -e "  • Video Server:     ${YELLOW}Waiting for roscore...${NC}"
    fi
fi
echo "  • Web Interface:    http://localhost:5174"
echo ""
echo "You can also access from other devices on your network:"
echo "  • http://$(hostname -I | awk '{print $1}'):5174"
echo ""
if [ "$ROSCORE_RUNNING" = false ]; then
    echo -e "${YELLOW}Note: Start roscore/simulation, then refresh the web page to connect${NC}"
    echo ""
fi
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for background processes
wait

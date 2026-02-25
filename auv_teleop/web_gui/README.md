# ITU AUV Web GUI

Web-based control interface for the ITU AUV system.

## Quick Start

### Option 1: One Command Startup (Recommended)

```bash
# Make the script executable (first time only)
chmod +x start_web_gui.sh

# Start everything
./start_web_gui.sh
```

This will automatically start:
- ROS Bridge (WebSocket server on port 9090)
- Web Video Server (Camera streaming on port 8080)
- Vite Dev Server (Web interface on port 5174)

Press `Ctrl+C` to stop all services.

### Option 2: Manual Startup

**Terminal 1 - Start ROS services:**
```bash
roslaunch auv_teleop web_gui.launch
```

**Terminal 2 - Start web server:**
```bash
cd /home/deniz/auv_ws/src/auv-software/auv_teleop/web_gui
npm install  # First time only
npm run dev
```

**Terminal 3 - Start video server (optional for cameras):**
```bash
rosrun web_video_server web_video_server
```

## Access the Interface

- **Local:** http://localhost:5174
- **Network:** http://YOUR_IP:5174 (shown in startup script output)

## Features

### Control Panel Tab
- 🔋 Battery status monitoring with power consumption graph
- 🌊 Depth control
- 🎮 Vehicle control (directional buttons)
- ⚙️ Services (Localization, DVL, Clear Objects, Reset Pose)
- 🎯 Mission controls (Torpedos, Ball Dropper)
- 👁️ Object detection controls
- 🤖 SMACH state machine controls

### Camera View Tab
- 📹 Live camera streaming
- 🎥 Dual camera view (front & bottom)
- 🔄 Auto-discovery of camera topics
- 📊 Topic selection dropdown

## Prerequisites

### ROS Packages
```bash
sudo apt-get install ros-noetic-rosbridge-server
sudo apt-get install ros-noetic-web-video-server
```

### Node.js (20.x LTS)
Already installed in this workspace.

## Development

### Project Structure
```
web_gui/
├── start_web_gui.sh          # Startup script
├── package.json              # NPM dependencies
├── vite.config.js           # Vite configuration
├── index.html               # Entry HTML (loads roslib.js)
└── src/
    ├── App.jsx              # Main application
    ├── theme.js             # Material-UI theme
    ├── components/          # React components
    │   ├── Header.jsx
    │   ├── BatteryStatus.jsx
    │   ├── DepthControl.jsx
    │   ├── VehicleControl.jsx
    │   ├── ServicesPanel.jsx
    │   ├── MissionControls.jsx
    │   ├── ObjectDetection.jsx
    │   ├── StateMachine.jsx
    │   └── CameraView.jsx
    ├── hooks/
    │   └── useROS.js        # ROS connection hook
    └── utils/
        └── rosServices.js   # ROS service utilities
```

### Tech Stack
- **Frontend:** React 18.2.0 + Vite 5.0.8
- **UI Library:** Material-UI 5.14.20
- **ROS Integration:** roslib.js 1.3.0 (CDN)
- **Build Tool:** Vite

## Troubleshooting

### "ROS master is not running"
Start roscore first:
```bash
roscore
```

### "Cannot connect to rosbridge"
Make sure rosbridge is running:
```bash
roslaunch auv_teleop web_gui.launch
```

### "No camera topics found"
1. Ensure camera nodes are running
2. Check if web_video_server is installed and running
3. Verify camera topics exist: `rostopic list | grep image`

### Port already in use
If ports 9090, 8080, or 5174 are already in use, you can modify them:
- ROS Bridge: Edit `launch/web_gui.launch`
- Web Video: Edit `launch/web_gui.launch`
- Vite: Edit `vite.config.js`

## Network Access

To access from other devices:
1. Make sure your firewall allows connections on ports 5174, 9090, and 8080
2. Find your IP: `hostname -I`
3. Open `http://YOUR_IP:5174` on another device

## Production Build

```bash
npm run build
```

This creates an optimized build in the `dist/` directory that can be served by any web server.

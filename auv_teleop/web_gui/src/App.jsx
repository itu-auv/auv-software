import React, { useState } from 'react';
//import ROSLIB from 'roslibjs';
//ROSLIB is loaded from CDN in index.html
import { Box, Alert, Tabs, Tab, Grid, Container } from '@mui/material';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Components
import Header from './components/Header';
import BatteryStatus from './components/BatteryStatus';
import DepthControl from './components/DepthControl';
import VehicleControl from './components/VehicleControl';
import ServicesPanel from './components/ServicesPanel';
import MissionControls from './components/MissionControls';
import ObjectDetection from './components/ObjectDetection';
import StateMachine from './components/StateMachine';
import CameraView from './components/CameraView';
import ThreeBackground from './components/ThreeBackground';
import VisualSettings from './components/VisualSettings';
import StartScreen from './components/StartScreen';

// Hooks and Utils
import { useROS } from './hooks/useROS';
import { callService, publishCmdVel, enableControl, disableControl } from './utils/rosServices';
import { createThemeByName } from './theme';

function App() {
  // Mode selection state
  const [mode, setMode] = useState(null); // null = start screen, 'pool' or 'simulation'
  
  // ROS Connection (from custom hook)
  const {
    ros,
    connected,
    connecting,
    connectToROS,
    voltage,
    current,
    power,
    powerHistory,
    powerTopicName,
    lastPowerUpdate,
    powerSource
  } = useROS();
  
  // UI State
  const [depth, setDepth] = useState(-1.0);
  const [detectionRunning, setDetectionRunning] = useState(false);
  const [smachRunning, setSmachRunning] = useState(false);
  const [testMode, setTestMode] = useState(false);
  const [cudaEnabled, setCudaEnabled] = useState(false);
  const [controlEnabled, setControlEnabled] = useState(false);
  const [activeTab, setActiveTab] = useState(0);
  const [selectedStates, setSelectedStates] = useState({
    init: false,
    gate: false,
    buoy: false,
    torpedo: false,
    bin: false,
    octagon: false,
  });

  // Visual Settings State
  const [threeEnabled, setThreeEnabled] = useState(true);
  const [currentTheme, setCurrentTheme] = useState('dark');
  const [fancyEffects, setFancyEffects] = useState(true);

  // Create theme dynamically based on theme name and fancy effects setting
  const theme = createThemeByName(currentTheme, fancyEffects);

  // Mode selection handler
  const handleSelectMode = (selectedMode) => {
    setMode(selectedMode);
    if (selectedMode === 'pool') {
      // Pool mode selected - will show message for now
      console.log('Pool mode selected - coming soon');
    }
  };

  // Show start screen if no mode selected
  if (!mode) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <StartScreen onSelectMode={handleSelectMode} />
      </ThemeProvider>
    );
  }

  // Show pool mode placeholder (for future implementation)
  if (mode === 'pool') {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            minHeight: '100vh',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%)',
          }}
        >
          <Container maxWidth="md" sx={{ textAlign: 'center' }}>
            <Box
              sx={{
                p: 4,
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(10px)',
                borderRadius: 4,
                border: '1px solid rgba(255, 255, 255, 0.1)',
              }}
            >
              <Box
                sx={{
                  fontSize: 80,
                  mb: 2,
                }}
              >
                🏊
              </Box>
              <Box
                sx={{
                  fontSize: 48,
                  fontWeight: 700,
                  mb: 2,
                  background: 'linear-gradient(135deg, #00D9FF 0%, #7C4DFF 100%)',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                Pool Test Mode
              </Box>
              <Box sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 4, fontSize: 18 }}>
                This mode will be available soon for real hardware testing
              </Box>
              <Box
                onClick={() => setMode(null)}
                sx={{
                  display: 'inline-block',
                  px: 4,
                  py: 1.5,
                  background: 'linear-gradient(135deg, #00D9FF 0%, #7C4DFF 100%)',
                  borderRadius: 2,
                  cursor: 'pointer',
                  fontWeight: 600,
                  '&:hover': {
                    transform: 'scale(1.05)',
                  },
                  transition: 'transform 0.2s',
                }}
              >
                ← Back to Mode Selection
              </Box>
            </Box>
          </Container>
        </Box>
      </ThemeProvider>
    );
  }

  // Service handlers
  const setDepthService = async () => {
    try {
      await callService(ros, '/taluy/set_depth', 'auv_msgs/SetDepth', { target_depth: depth });
      console.log('Depth set to:', depth);
    } catch (error) {
      console.error('Failed to set depth:', error);
    }
  };

  const startLocalization = () => callService(ros, '/taluy/auv_localization_node/enable', 'std_srvs/Empty');
  const handleEnableDVL = () => callService(ros, '/taluy/sensors/dvl/enable', 'std_srvs/SetBool', { data: true });
  const handleDisableDVL = () => callService(ros, '/taluy/sensors/dvl/enable', 'std_srvs/SetBool', { data: false });
  const clearObjects = () => callService(ros, '/taluy/map/clear_object_transforms', 'std_srvs/Empty');
  const resetPose = () => callService(ros, '/taluy/reset_odometry', 'std_srvs/Empty');
  const launchTorpedo1 = () => callService(ros, '/taluy/actuators/torpedo_1/launch', 'std_srvs/Trigger');
  const launchTorpedo2 = () => callService(ros, '/taluy/actuators/torpedo_2/launch', 'std_srvs/Trigger');
  const dropBall = () => callService(ros, '/taluy/actuators/ball_dropper/drop', 'std_srvs/Trigger');

  // Vehicle control handlers
  const handlePublishCmdVel = (linear, angular) => publishCmdVel(ros, linear, angular);
  const stopVehicle = () => handlePublishCmdVel({ x: 0, y: 0, z: 0 }, { z: 0 });
  const handleEnableControl = () => enableControl(ros, setControlEnabled);
  const handleDisableControl = () => disableControl(ros, setControlEnabled);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      {threeEnabled && <ThreeBackground enabled={threeEnabled} />}
      <VisualSettings
        threeEnabled={threeEnabled}
        setThreeEnabled={setThreeEnabled}
        currentTheme={currentTheme}
        setTheme={setCurrentTheme}
        fancyEffects={fancyEffects}
        setFancyEffects={setFancyEffects}
      />
      <Box sx={{ minHeight: '100vh', bgcolor: 'transparent', width: '100vw', maxWidth: '100vw', overflow: 'auto', position: 'relative', zIndex: 1 }}>
        <Header 
          connected={connected} 
          connecting={connecting} 
          connectToROS={connectToROS}
          fancyEffects={fancyEffects}
        />

        {!connected && (
          <Box sx={{ pt: 2, px: 3 }}>
            <Alert severity="warning">
              Not connected to ROS. Make sure rosbridge_server is running: 
              <code style={{ marginLeft: 8 }}>roslaunch auv_teleop web_gui.launch</code>
            </Alert>
          </Box>
        )}

        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'background.paper', width: '100%' }}>
          <Box sx={{ px: 3, width: '100%' }}>
            <Tabs 
              value={activeTab} 
              onChange={(e, newValue) => setActiveTab(newValue)}
              textColor="primary"
              indicatorColor="primary"
              sx={{ minHeight: 64, width: '100%' }}
            >
              <Tab 
                label="🎮 Control Panel" 
                sx={{ fontSize: '1rem', minHeight: 64, fontWeight: 600 }}
              />
              <Tab 
                label="📹 Camera View" 
                sx={{ fontSize: '1rem', minHeight: 64, fontWeight: 600 }}
              />
            </Tabs>
          </Box>
        </Box>

        <Box sx={{ py: 4, width: '100%' }}>
          {/* Control Panel Tab */}
          {activeTab === 0 && (
            <Box sx={{ px: 3 }}>
            <Grid container spacing={3}>
              {/* Row 1: Status & Monitoring */}
              <Grid item xs={12} md={6} lg={3}>
                <BatteryStatus
                  voltage={voltage}
                  current={current}
                  power={power}
                  powerHistory={powerHistory}
                  powerTopicName={powerTopicName}
                  lastPowerUpdate={lastPowerUpdate}
                  powerSource={powerSource}
                />
              </Grid>

              <Grid item xs={12} md={6} lg={3}>
                <DepthControl
                  depth={depth}
                  setDepth={setDepth}
                  setDepthService={setDepthService}
                  connected={connected}
                />
              </Grid>

              <Grid item xs={12} md={6} lg={3}>
                <ServicesPanel
                  connected={connected}
                  startLocalization={startLocalization}
                  enableDVL={handleEnableDVL}
                  disableDVL={handleDisableDVL}
                  clearObjects={clearObjects}
                  resetPose={resetPose}
                />
              </Grid>

              <Grid item xs={12} md={6} lg={3}>
                <MissionControls
                  connected={connected}
                  launchTorpedo1={launchTorpedo1}
                  launchTorpedo2={launchTorpedo2}
                  dropBall={dropBall}
                />
              </Grid>

              {/* Row 2: Vehicle Control & Detection */}
              <Grid item xs={12} md={6}>
                <VehicleControl
                  connected={connected}
                  controlEnabled={controlEnabled}
                  enableControl={handleEnableControl}
                  disableControl={handleDisableControl}
                  publishCmdVel={handlePublishCmdVel}
                  stopVehicle={stopVehicle}
                  ros={ros}
                  fancyEffects={fancyEffects}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <ObjectDetection
                  connected={connected}
                  ros={ros}
                  cudaEnabled={cudaEnabled}
                  setCudaEnabled={setCudaEnabled}
                  detectionRunning={detectionRunning}
                  setDetectionRunning={setDetectionRunning}
                />
              </Grid>

              {/* Row 3: Full-width State Machine */}
              <Grid item xs={12}>
                <StateMachine
                  connected={connected}
                  testMode={testMode}
                  setTestMode={setTestMode}
                  smachRunning={smachRunning}
                  setSmachRunning={setSmachRunning}
                  selectedStates={selectedStates}
                  setSelectedStates={setSelectedStates}
                />
              </Grid>
            </Grid>
            </Box>
          )}

          {/* Camera View Tab */}
          {activeTab === 1 && (
            <CameraView connected={connected} ros={ros} />
          )}
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
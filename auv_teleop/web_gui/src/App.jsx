import React, { useState } from 'react';

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
import VisualSettings from './components/VisualSettings';
import StartScreen from './components/StartScreen';
import GamepadVisualization from './components/GamepadVisualization';

// UI Components
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';

// Hooks and Utils
import { useROS } from './hooks/useROS';
import { callService, publishCmdVel, enableControl, disableControl } from './utils/rosServices';

// Icons
import { Gamepad2, Camera, AlertTriangle, ArrowLeft } from 'lucide-react';

function App() {
  // Mode selection state
  const [mode, setMode] = useState(null);

  // ROS Connection
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
  const [teleopRunning, setTeleopRunning] = useState(false);
  const [activeTab, setActiveTab] = useState('control');
  const [selectedStates, setSelectedStates] = useState({
    init: false,
    gate: false,
    buoy: false,
    torpedo: false,
    bin: false,
    octagon: false,
  });

  // Visual Settings State
  const [currentTheme, setCurrentTheme] = useState('dark');
  const [fancyEffects, setFancyEffects] = useState(true);

  // Mode selection handler
  const handleSelectMode = (selectedMode) => {
    setMode(selectedMode);
  };

  // Show start screen if no mode selected
  if (!mode) {
    return (
      <div className={currentTheme === 'halloween' ? 'theme-halloween' : ''}>
        <VisualSettings
          currentTheme={currentTheme}
          setTheme={setCurrentTheme}
          fancyEffects={fancyEffects}
          setFancyEffects={setFancyEffects}
        />
        <StartScreen onSelectMode={handleSelectMode} />
      </div>
    );
  }

  // Pool mode placeholder
  if (mode === 'pool') {
    return (
      <div className={`min-h-screen flex items-center justify-center bg-black ${currentTheme === 'halloween' ? 'theme-halloween' : ''}`}>
        <VisualSettings
          currentTheme={currentTheme}
          setTheme={setCurrentTheme}
          fancyEffects={fancyEffects}
          setFancyEffects={setFancyEffects}
        />
        <Card className="max-w-md">
          <CardContent className="pt-6 text-center">
            <div className="text-7xl mb-4">🏊</div>
            <h1 className="text-3xl font-bold mb-4 text-white">
              Pool Test Mode
            </h1>
            <p className="text-white/50 mb-6">
              This mode will be available soon for real hardware testing
            </p>
            <Button onClick={() => setMode(null)} variant="outline">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Mode Selection
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Service handlers
  const setDepthService = async () => {
    try {
      await callService(ros, '/taluy/set_depth', 'auv_msgs/SetDepth', { target_depth: depth });
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
    <div className={`min-h-screen bg-black ${currentTheme === 'halloween' ? 'theme-halloween' : ''}`}>
      <VisualSettings
        currentTheme={currentTheme}
        setTheme={setCurrentTheme}
        fancyEffects={fancyEffects}
        setFancyEffects={setFancyEffects}
      />

      <Header
        connected={connected}
        connecting={connecting}
        connectToROS={connectToROS}
        fancyEffects={fancyEffects}
      />

      <main className="container mx-auto px-6 py-8">
        {!connected && (
          <Alert variant="warning" className="mb-8">
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Not connected to ROS. Make sure rosbridge_server is running:{' '}
              <code className="ml-2 bg-white/5 px-2 py-1 rounded-lg text-xs font-mono">
                roslaunch auv_teleop web_gui.launch
              </code>
            </AlertDescription>
          </Alert>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-lg grid-cols-3 mb-8">
            <TabsTrigger value="control" className="gap-2">
              <Gamepad2 className="w-4 h-4" />
              Control
            </TabsTrigger>
            <TabsTrigger value="gamepad" className="gap-2">
              <Gamepad2 className="w-4 h-4" />
              Gamepad
            </TabsTrigger>
            <TabsTrigger value="camera" className="gap-2">
              <Camera className="w-4 h-4" />
              Camera
            </TabsTrigger>
          </TabsList>

          <TabsContent value="control" className="space-y-6">
            {/* First Row - Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <BatteryStatus
                voltage={voltage}
                current={current}
                power={power}
                powerHistory={powerHistory}
                powerTopicName={powerTopicName}
                lastPowerUpdate={lastPowerUpdate}
                powerSource={powerSource}
              />

              <DepthControl
                depth={depth}
                setDepth={setDepth}
                setDepthService={setDepthService}
                connected={connected}
              />

              <ServicesPanel
                connected={connected}
                startLocalization={startLocalization}
                enableDVL={handleEnableDVL}
                disableDVL={handleDisableDVL}
                clearObjects={clearObjects}
                resetPose={resetPose}
              />

              <MissionControls
                connected={connected}
                launchTorpedo1={launchTorpedo1}
                launchTorpedo2={launchTorpedo2}
                dropBall={dropBall}
              />
            </div>

            {/* Second Row - Control Panels */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <VehicleControl
                connected={connected}
                controlEnabled={controlEnabled}
                enableControl={handleEnableControl}
                disableControl={handleDisableControl}
                publishCmdVel={handlePublishCmdVel}
                stopVehicle={stopVehicle}
                ros={ros}
                fancyEffects={fancyEffects}
                teleopRunning={teleopRunning}
                setTeleopRunning={setTeleopRunning}
              />

              <ObjectDetection
                connected={connected}
                ros={ros}
                cudaEnabled={cudaEnabled}
                setCudaEnabled={setCudaEnabled}
                detectionRunning={detectionRunning}
                setDetectionRunning={setDetectionRunning}
              />
            </div>

            {/* Third Row - State Machine */}
            <StateMachine
              connected={connected}
              testMode={testMode}
              setTestMode={setTestMode}
              smachRunning={smachRunning}
              setSmachRunning={setSmachRunning}
              selectedStates={selectedStates}
              setSelectedStates={setSelectedStates}
            />
          </TabsContent>

          <TabsContent value="gamepad">
            <div className="max-w-2xl mx-auto">
              <GamepadVisualization ros={ros} connected={connected} />
            </div>
          </TabsContent>

          <TabsContent value="camera">
            <CameraView connected={connected} ros={ros} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}

export default App;

import React, { useRef, useState } from 'react';
import {
  ArrowUp, ArrowDown, ArrowLeft, ArrowRight,
  StopCircle, ChevronUp, ChevronDown,
  RotateCcw, RotateCw, Gamepad2,
  Play, Square
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';

function VehicleControl({
  connected,
  controlEnabled,
  enableControl,
  disableControl,
  publishCmdVel,
  stopVehicle,
  ros,
  teleopRunning,
  setTeleopRunning
}) {
  const publishIntervalRef = useRef(null);
  const [linearSpeed, setLinearSpeed] = useState(0.2);
  const [angularSpeed, setAngularSpeed] = useState(0.2);
  const [xboxMode, setXboxMode] = useState(false);
  const [deviceId, setDeviceId] = useState(1);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const startPublishing = (linear, angular) => {
    if (publishIntervalRef.current) {
      clearInterval(publishIntervalRef.current);
    }
    publishCmdVel(linear, angular);
    publishIntervalRef.current = setInterval(() => {
      publishCmdVel(linear, angular);
    }, 50);
  };

  const stopPublishing = () => {
    if (publishIntervalRef.current) {
      clearInterval(publishIntervalRef.current);
      publishIntervalRef.current = null;
    }
    stopVehicle();
  };

  const handleStartTeleop = async () => {
    if (!ros || !ros.isConnected) {
      setSnackbar({ open: true, message: 'ROS not connected!', severity: 'error' });
      return;
    }

    ros.getServices((services) => {
      if (!services.includes('/taluy/web_gui/start_teleop')) {
        setSnackbar({ open: true, message: 'Start teleop service not available.', severity: 'error' });
        return;
      }

      try {
        const paramClient = new window.ROSLIB.Param({
          ros: ros,
          name: '/web_gui_teleop_service/device_id'
        });
        paramClient.set(deviceId);

        const service = new window.ROSLIB.Service({
          ros: ros,
          name: '/taluy/web_gui/start_teleop',
          serviceType: 'std_srvs/SetBool'
        });

        service.callService(
          new window.ROSLIB.ServiceRequest({ data: xboxMode }),
          (result) => {
            if (result.success) {
              setTeleopRunning(true);
              setSnackbar({ open: true, message: `Teleop started!`, severity: 'success' });
            } else {
              setSnackbar({ open: true, message: `Failed: ${result.message}`, severity: 'error' });
            }
          },
          (error) => setSnackbar({ open: true, message: `Service failed: ${error}`, severity: 'error' })
        );
      } catch (error) {
        setSnackbar({ open: true, message: `Error: ${error}`, severity: 'error' });
      }
    });
  };

  const handleStopTeleop = async () => {
    if (!ros || !ros.isConnected) return;

    try {
      const service = new window.ROSLIB.Service({
        ros: ros,
        name: '/taluy/web_gui/stop_teleop',
        serviceType: 'std_srvs/Trigger'
      });

      service.callService(
        new window.ROSLIB.ServiceRequest({}),
        () => {
          setTeleopRunning(false);
          setSnackbar({ open: true, message: 'Teleop stopped', severity: 'success' });
        },
        () => setTeleopRunning(false)
      );
    } catch {
      setTeleopRunning(false);
    }
  };

  const ControlButton = ({ onPress, disabled, children, className = '', variant = 'default' }) => (
    <button
      className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 disabled:opacity-30 border ${
        variant === 'default'
          ? 'bg-white/[0.03] hover:bg-white/[0.08] border-white/[0.08] text-white/70 hover:text-white'
          : variant === 'secondary'
          ? 'bg-white/[0.05] hover:bg-white/[0.10] border-white/[0.10] text-white/60 hover:text-white'
          : 'bg-red-500/10 hover:bg-red-500/20 border-red-500/30 text-red-400'
      } ${className}`}
      disabled={disabled}
      onMouseDown={onPress ? () => onPress() : undefined}
      onMouseUp={stopPublishing}
      onMouseLeave={stopPublishing}
      onTouchStart={onPress ? () => onPress() : undefined}
      onTouchEnd={stopPublishing}
    >
      {children}
    </button>
  );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Gamepad2 className="w-4 h-4 text-white/50" />
            Vehicle Control
          </span>
          <span className={`text-xs font-medium px-2 py-1 rounded-full ${
            controlEnabled
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-white/5 text-white/30 border border-white/10'
          }`}>
            {controlEnabled ? 'Enabled' : 'Disabled'}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Teleoperation Section */}
        <div className="p-4 rounded-xl border border-white/[0.06] bg-white/[0.02] space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-white/70">Joystick Teleop</span>
            <code className="text-[10px] text-white/30 font-mono">/dev/input/js{deviceId}</code>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="success"
              onClick={handleStartTeleop}
              disabled={!connected || teleopRunning}
            >
              <Play className="w-3 h-3 mr-1" />
              Start
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={handleStopTeleop}
              disabled={!connected || !teleopRunning}
            >
              <Square className="w-3 h-3 mr-1" />
              Stop
            </Button>

            <div className="flex items-center gap-2 ml-auto">
              <span className="text-[10px] text-white/30">Xbox</span>
              <Switch
                checked={xboxMode}
                onCheckedChange={setXboxMode}
                disabled={teleopRunning}
              />
            </div>

            <select
              value={deviceId}
              onChange={(e) => setDeviceId(parseInt(e.target.value))}
              disabled={teleopRunning}
              className="bg-white/[0.03] border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white/70 focus:outline-none focus:border-white/30"
            >
              <option value={0}>js0</option>
              <option value={1}>js1</option>
              <option value={2}>js2</option>
              <option value={3}>js3</option>
            </select>
          </div>
        </div>

        {/* Control Enable/Disable */}
        <div className="flex gap-2 justify-center">
          <Button
            size="sm"
            variant="success"
            onClick={enableControl}
            disabled={!connected || controlEnabled}
          >
            Enable Control
          </Button>
          <Button
            size="sm"
            variant="secondary"
            onClick={disableControl}
            disabled={!connected || !controlEnabled}
          >
            Disable Control
          </Button>
        </div>

        {/* Control Buttons */}
        <div className="flex gap-8 justify-center pt-2">
          {/* Translation Controls */}
          <div className="flex flex-col items-center gap-1">
            <span className="text-[10px] uppercase tracking-wider text-white/30 mb-2">Translation</span>
            <ControlButton
              onPress={() => startPublishing({ x: linearSpeed, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })}
              disabled={!connected}
            >
              <ArrowUp className="w-4 h-4" />
            </ControlButton>

            <div className="flex gap-1">
              <ControlButton
                onPress={() => startPublishing({ x: 0, y: linearSpeed, z: 0 }, { x: 0, y: 0, z: 0 })}
                disabled={!connected}
              >
                <ArrowLeft className="w-4 h-4" />
              </ControlButton>

              <ControlButton
                variant="error"
                disabled={!connected}
                onPress={stopPublishing}
              >
                <StopCircle className="w-4 h-4" />
              </ControlButton>

              <ControlButton
                onPress={() => startPublishing({ x: 0, y: -linearSpeed, z: 0 }, { x: 0, y: 0, z: 0 })}
                disabled={!connected}
              >
                <ArrowRight className="w-4 h-4" />
              </ControlButton>
            </div>

            <ControlButton
              onPress={() => startPublishing({ x: -linearSpeed, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })}
              disabled={!connected}
            >
              <ArrowDown className="w-4 h-4" />
            </ControlButton>

            {/* Linear Speed */}
            <div className="w-36 mt-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-white/30">Linear</span>
                <span className="text-[10px] font-mono text-white/50">{linearSpeed.toFixed(2)}</span>
              </div>
              <Slider
                value={[linearSpeed]}
                onValueChange={([v]) => setLinearSpeed(v)}
                min={0.1}
                max={1.0}
                step={0.05}
                disabled={!connected}
              />
            </div>
          </div>

          {/* Depth & Rotation Controls */}
          <div className="flex flex-col items-center gap-1">
            <span className="text-[10px] uppercase tracking-wider text-white/30 mb-2">Depth/Rotation</span>
            <ControlButton
              variant="secondary"
              onPress={() => startPublishing({ x: 0, y: 0, z: linearSpeed }, { x: 0, y: 0, z: 0 })}
              disabled={!connected}
            >
              <ChevronUp className="w-4 h-4" />
            </ControlButton>

            <div className="flex gap-1">
              <ControlButton
                variant="secondary"
                onPress={() => startPublishing({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: angularSpeed })}
                disabled={!connected}
              >
                <RotateCcw className="w-4 h-4" />
              </ControlButton>

              <ControlButton
                variant="secondary"
                onPress={() => startPublishing({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -angularSpeed })}
                disabled={!connected}
              >
                <RotateCw className="w-4 h-4" />
              </ControlButton>
            </div>

            <ControlButton
              variant="secondary"
              onPress={() => startPublishing({ x: 0, y: 0, z: -linearSpeed }, { x: 0, y: 0, z: 0 })}
              disabled={!connected}
            >
              <ChevronDown className="w-4 h-4" />
            </ControlButton>

            {/* Angular Speed */}
            <div className="w-36 mt-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-white/30">Angular</span>
                <span className="text-[10px] font-mono text-white/50">{angularSpeed.toFixed(2)}</span>
              </div>
              <Slider
                value={[angularSpeed]}
                onValueChange={([v]) => setAngularSpeed(v)}
                min={0.1}
                max={1.0}
                step={0.05}
                disabled={!connected}
              />
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default VehicleControl;

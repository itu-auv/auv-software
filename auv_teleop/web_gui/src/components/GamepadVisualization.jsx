import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Gamepad2, Wifi, WifiOff, Settings2 } from 'lucide-react';

function GamepadVisualization({ ros, connected }) {
  const [gamepadConnected, setGamepadConnected] = useState(false);
  const [gamepadName, setGamepadName] = useState('');
  const [axes, setAxes] = useState([0, 0, 0, 0, 0, 0]);
  const [buttons, setButtons] = useState(new Array(17).fill(false));
  const [showBindings, setShowBindings] = useState(false);
  const [controllerType, setControllerType] = useState('xbox'); // 'xbox' or 'ps'
  const [deadzone, setDeadzone] = useState(0.1);
  const [vibrationEnabled, setVibrationEnabled] = useState(true);

  // Button mappings for AUV control
  const buttonMappings = {
    xbox: {
      0: 'A - Drop Ball',
      1: 'B - Torpedo 2',
      2: 'X - Torpedo 1',
      3: 'Y',
      4: 'LB - Z Down',
      5: 'RB - Z Up',
      6: 'Back',
      7: 'Start',
      8: 'Xbox',
      9: 'L Stick',
      10: 'R Stick',
      11: 'D-Up',
      12: 'D-Down',
      13: 'D-Left',
      14: 'D-Right',
    },
    ps: {
      0: 'X - Drop Ball',
      1: 'O - Torpedo 2',
      2: '□ - Torpedo 1',
      3: '△',
      4: 'L1 - Z Down',
      5: 'R1 - Z Up',
      6: 'Share',
      7: 'Options',
      8: 'PS',
      9: 'L3',
      10: 'R3',
      11: 'D-Up',
      12: 'D-Down',
      13: 'D-Left',
      14: 'D-Right',
    }
  };

  const axisMappings = {
    0: 'Left Stick X (Y-axis / Strafe)',
    1: 'Left Stick Y (X-axis / Forward)',
    2: 'Right Stick X (Yaw)',
    3: 'Right Stick Y',
    4: 'Left Trigger (Z Down)',
    5: 'Right Trigger (Z Up)',
    6: 'D-Pad X',
    7: 'D-Pad Y',
  };

  // Gamepad polling
  const pollGamepad = useCallback(() => {
    const gamepads = navigator.getGamepads();
    const gp = gamepads[0] || gamepads[1] || gamepads[2] || gamepads[3];

    if (gp) {
      setGamepadConnected(true);
      setGamepadName(gp.id);

      // Detect controller type
      if (gp.id.toLowerCase().includes('xbox') || gp.id.toLowerCase().includes('microsoft')) {
        setControllerType('xbox');
      } else if (gp.id.toLowerCase().includes('playstation') || gp.id.toLowerCase().includes('sony') || gp.id.toLowerCase().includes('dualshock') || gp.id.toLowerCase().includes('dualsense')) {
        setControllerType('ps');
      }

      // Apply deadzone
      const processedAxes = gp.axes.map(axis =>
        Math.abs(axis) < deadzone ? 0 : axis
      );
      setAxes(processedAxes);
      setButtons(gp.buttons.map(btn => btn.pressed));
    } else {
      setGamepadConnected(false);
      setGamepadName('');
      setAxes([0, 0, 0, 0, 0, 0]);
      setButtons(new Array(17).fill(false));
    }
  }, [deadzone]);

  useEffect(() => {
    const interval = setInterval(pollGamepad, 16); // ~60fps

    const handleConnect = (e) => {
      console.log('Gamepad connected:', e.gamepad.id);
      if (vibrationEnabled && e.gamepad.vibrationActuator) {
        e.gamepad.vibrationActuator.playEffect('dual-rumble', {
          duration: 200,
          strongMagnitude: 0.5,
          weakMagnitude: 0.5
        });
      }
    };

    const handleDisconnect = () => {
      console.log('Gamepad disconnected');
      setGamepadConnected(false);
    };

    window.addEventListener('gamepadconnected', handleConnect);
    window.addEventListener('gamepaddisconnected', handleDisconnect);

    return () => {
      clearInterval(interval);
      window.removeEventListener('gamepadconnected', handleConnect);
      window.removeEventListener('gamepaddisconnected', handleDisconnect);
    };
  }, [pollGamepad, vibrationEnabled]);

  // SVG Controller visualization
  const ControllerSVG = () => {
    const isXbox = controllerType === 'xbox';

    return (
      <svg viewBox="0 0 400 220" className="w-full max-w-lg mx-auto">
        {/* Controller body */}
        <defs>
          <linearGradient id="bodyGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="rgba(255,255,255,0.08)" />
            <stop offset="100%" stopColor="rgba(255,255,255,0.02)" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>

        {/* Main body */}
        <path
          d="M60,100 Q60,60 100,50 L150,45 Q200,40 250,45 L300,50 Q340,60 340,100 L350,140 Q360,180 320,190 L280,195 Q240,200 200,200 Q160,200 120,195 L80,190 Q40,180 50,140 Z"
          fill="url(#bodyGradient)"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="2"
        />

        {/* Grips */}
        <ellipse cx="70" cy="160" rx="35" ry="50" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" />
        <ellipse cx="330" cy="160" rx="35" ry="50" fill="rgba(255,255,255,0.03)" stroke="rgba(255,255,255,0.08)" />

        {/* Left Stick Background */}
        <circle cx="120" cy="110" r="28" fill="rgba(0,0,0,0.4)" stroke="rgba(255,255,255,0.1)" />
        {/* Left Stick */}
        <circle
          cx={120 + axes[0] * 15}
          cy={110 + axes[1] * 15}
          r="20"
          fill={buttons[9] ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.15)'}
          stroke="rgba(255,255,255,0.3)"
          strokeWidth="2"
          filter={buttons[9] ? 'url(#glow)' : undefined}
        />

        {/* Right Stick Background */}
        <circle cx="250" cy="150" r="28" fill="rgba(0,0,0,0.4)" stroke="rgba(255,255,255,0.1)" />
        {/* Right Stick */}
        <circle
          cx={250 + axes[2] * 15}
          cy={150 + (axes[3] || 0) * 15}
          r="20"
          fill={buttons[10] ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.15)'}
          stroke="rgba(255,255,255,0.3)"
          strokeWidth="2"
          filter={buttons[10] ? 'url(#glow)' : undefined}
        />

        {/* D-Pad */}
        <g transform="translate(165, 145)">
          {/* Up */}
          <rect x="-8" y="-28" width="16" height="18" rx="2"
            fill={buttons[12] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.1)'}
            stroke="rgba(255,255,255,0.2)"
          />
          {/* Down */}
          <rect x="-8" y="10" width="16" height="18" rx="2"
            fill={buttons[13] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.1)'}
            stroke="rgba(255,255,255,0.2)"
          />
          {/* Left */}
          <rect x="-28" y="-8" width="18" height="16" rx="2"
            fill={buttons[14] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.1)'}
            stroke="rgba(255,255,255,0.2)"
          />
          {/* Right */}
          <rect x="10" y="-8" width="18" height="16" rx="2"
            fill={buttons[15] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.1)'}
            stroke="rgba(255,255,255,0.2)"
          />
          {/* Center */}
          <rect x="-8" y="-8" width="16" height="16" rx="2"
            fill="rgba(255,255,255,0.05)"
          />
        </g>

        {/* Face Buttons */}
        <g transform="translate(300, 100)">
          {/* Y / Triangle - Top */}
          <circle cx="0" cy="-22" r="12"
            fill={buttons[3] ? (isXbox ? 'rgba(255,200,0,0.6)' : 'rgba(0,255,150,0.6)') : 'rgba(255,255,255,0.1)'}
            stroke={isXbox ? 'rgba(255,200,0,0.4)' : 'rgba(0,255,150,0.4)'}
            strokeWidth="2"
            filter={buttons[3] ? 'url(#glow)' : undefined}
          />
          <text x="0" y="-18" textAnchor="middle" fontSize="10" fill="rgba(255,255,255,0.7)">
            {isXbox ? 'Y' : '△'}
          </text>

          {/* A / Cross - Bottom */}
          <circle cx="0" cy="22" r="12"
            fill={buttons[0] ? (isXbox ? 'rgba(0,255,0,0.6)' : 'rgba(100,150,255,0.6)') : 'rgba(255,255,255,0.1)'}
            stroke={isXbox ? 'rgba(0,255,0,0.4)' : 'rgba(100,150,255,0.4)'}
            strokeWidth="2"
            filter={buttons[0] ? 'url(#glow)' : undefined}
          />
          <text x="0" y="26" textAnchor="middle" fontSize="10" fill="rgba(255,255,255,0.7)">
            {isXbox ? 'A' : 'X'}
          </text>

          {/* X / Square - Left */}
          <circle cx="-22" cy="0" r="12"
            fill={buttons[2] ? (isXbox ? 'rgba(0,150,255,0.6)' : 'rgba(255,100,150,0.6)') : 'rgba(255,255,255,0.1)'}
            stroke={isXbox ? 'rgba(0,150,255,0.4)' : 'rgba(255,100,150,0.4)'}
            strokeWidth="2"
            filter={buttons[2] ? 'url(#glow)' : undefined}
          />
          <text x="-22" y="4" textAnchor="middle" fontSize="10" fill="rgba(255,255,255,0.7)">
            {isXbox ? 'X' : '□'}
          </text>

          {/* B / Circle - Right */}
          <circle cx="22" cy="0" r="12"
            fill={buttons[1] ? (isXbox ? 'rgba(255,0,0,0.6)' : 'rgba(255,100,100,0.6)') : 'rgba(255,255,255,0.1)'}
            stroke={isXbox ? 'rgba(255,0,0,0.4)' : 'rgba(255,100,100,0.4)'}
            strokeWidth="2"
            filter={buttons[1] ? 'url(#glow)' : undefined}
          />
          <text x="22" y="4" textAnchor="middle" fontSize="10" fill="rgba(255,255,255,0.7)">
            {isXbox ? 'B' : 'O'}
          </text>
        </g>

        {/* Bumpers */}
        {/* LB */}
        <rect x="85" y="35" width="50" height="15" rx="5"
          fill={buttons[4] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.1)'}
          stroke="rgba(255,255,255,0.2)"
          filter={buttons[4] ? 'url(#glow)' : undefined}
        />
        <text x="110" y="46" textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.6)">LB</text>

        {/* RB */}
        <rect x="265" y="35" width="50" height="15" rx="5"
          fill={buttons[5] ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.1)'}
          stroke="rgba(255,255,255,0.2)"
          filter={buttons[5] ? 'url(#glow)' : undefined}
        />
        <text x="290" y="46" textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.6)">RB</text>

        {/* Triggers (shown as bars) */}
        {/* LT */}
        <rect x="85" y="15" width="50" height="12" rx="3" fill="rgba(0,0,0,0.3)" stroke="rgba(255,255,255,0.1)" />
        <rect x="85" y="15" width={50 * Math.max(0, axes[4] !== undefined ? (axes[4] + 1) / 2 : 0)} height="12" rx="3"
          fill="rgba(255,100,100,0.6)"
        />
        <text x="110" y="24" textAnchor="middle" fontSize="7" fill="rgba(255,255,255,0.6)">LT</text>

        {/* RT */}
        <rect x="265" y="15" width="50" height="12" rx="3" fill="rgba(0,0,0,0.3)" stroke="rgba(255,255,255,0.1)" />
        <rect x="265" y="15" width={50 * Math.max(0, axes[5] !== undefined ? (axes[5] + 1) / 2 : 0)} height="12" rx="3"
          fill="rgba(100,255,100,0.6)"
        />
        <text x="290" y="24" textAnchor="middle" fontSize="7" fill="rgba(255,255,255,0.6)">RT</text>

        {/* Center buttons */}
        {/* Back/Select */}
        <rect x="155" y="80" width="25" height="10" rx="3"
          fill={buttons[6] ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}
          stroke="rgba(255,255,255,0.15)"
        />
        {/* Start */}
        <rect x="220" y="80" width="25" height="10" rx="3"
          fill={buttons[7] ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.1)'}
          stroke="rgba(255,255,255,0.15)"
        />
        {/* Xbox/PS button */}
        <circle cx="200" cy="85" r="12"
          fill={buttons[8] ? 'rgba(255,255,255,0.4)' : 'rgba(255,255,255,0.08)'}
          stroke="rgba(255,255,255,0.2)"
        />
      </svg>
    );
  };

  // Axis visualization bar
  const AxisBar = ({ value, label, color = 'white' }) => {
    const percentage = ((value + 1) / 2) * 100;
    const isNeutral = Math.abs(value) < deadzone;

    return (
      <div className="flex items-center gap-3">
        <span className="text-xs text-white/40 w-24 truncate">{label}</span>
        <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden relative">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-px h-full bg-white/20" />
          </div>
          <div
            className="h-full transition-all duration-75 rounded-full"
            style={{
              width: `${Math.abs(value) * 50}%`,
              marginLeft: value < 0 ? `${50 - Math.abs(value) * 50}%` : '50%',
              backgroundColor: isNeutral ? 'rgba(255,255,255,0.1)' : `rgba(255,255,255,0.5)`
            }}
          />
        </div>
        <span className={`text-xs font-mono w-12 text-right ${isNeutral ? 'text-white/30' : 'text-white/70'}`}>
          {value.toFixed(2)}
        </span>
      </div>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Gamepad2 className="w-4 h-4 text-white/50" />
            Gamepad
          </span>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowBindings(!showBindings)}
              className={`p-1.5 rounded-lg transition-colors ${showBindings ? 'bg-white/10 text-white' : 'text-white/40 hover:text-white/60'}`}
            >
              <Settings2 className="w-4 h-4" />
            </button>
            <span className={`flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded-full ${
              gamepadConnected
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                : 'bg-white/5 text-white/30 border border-white/10'
            }`}>
              {gamepadConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              {gamepadConnected ? 'Connected' : 'No Gamepad'}
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Controller name */}
        {gamepadConnected && (
          <div className="text-xs text-white/40 text-center truncate px-4">
            {gamepadName}
          </div>
        )}

        {/* Controller SVG */}
        <div className="py-2">
          <ControllerSVG />
        </div>

        {/* Axes display */}
        <div className="space-y-2 pt-2 border-t border-white/5">
          <div className="text-xs text-white/50 font-medium mb-3">Axes</div>
          <div className="grid grid-cols-1 gap-2">
            <AxisBar value={axes[0] || 0} label="Left X (Strafe)" />
            <AxisBar value={axes[1] || 0} label="Left Y (Forward)" />
            <AxisBar value={axes[2] || 0} label="Right X (Yaw)" />
            <AxisBar value={axes[3] || 0} label="Right Y" />
          </div>
        </div>

        {/* Settings panel */}
        {showBindings && (
          <div className="space-y-4 pt-4 border-t border-white/5">
            <div className="text-xs text-white/50 font-medium">Settings</div>

            {/* Deadzone */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-white/60">Deadzone</span>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min="0"
                  max="0.3"
                  step="0.01"
                  value={deadzone}
                  onChange={(e) => setDeadzone(parseFloat(e.target.value))}
                  className="w-24 accent-white"
                />
                <span className="text-xs text-white/40 w-10">{deadzone.toFixed(2)}</span>
              </div>
            </div>

            {/* Vibration */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-white/60">Vibration</span>
              <Switch
                checked={vibrationEnabled}
                onCheckedChange={setVibrationEnabled}
              />
            </div>

            {/* Button bindings */}
            <div className="space-y-2">
              <div className="text-xs text-white/50 font-medium">AUV Button Bindings</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Torpedo 1</span>
                  <span className="text-white/70">{controllerType === 'xbox' ? 'X' : '□'}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Torpedo 2</span>
                  <span className="text-white/70">{controllerType === 'xbox' ? 'B' : 'O'}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Drop Ball</span>
                  <span className="text-white/70">{controllerType === 'xbox' ? 'A' : 'X'}</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Z Up</span>
                  <span className="text-white/70">RB / RT</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Z Down</span>
                  <span className="text-white/70">LB / LT</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Forward</span>
                  <span className="text-white/70">Left Stick Y</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Strafe</span>
                  <span className="text-white/70">Left Stick X</span>
                </div>
                <div className="flex justify-between p-2 rounded bg-white/[0.02] border border-white/5">
                  <span className="text-white/40">Yaw</span>
                  <span className="text-white/70">Right Stick X</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Connection hint */}
        {!gamepadConnected && (
          <div className="text-center py-4">
            <p className="text-xs text-white/30">
              Press any button on your controller to connect
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default GamepadVisualization;

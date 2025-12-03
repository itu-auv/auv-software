import React, { useRef, useState } from 'react';
import { Box, Card, CardContent, Typography, Button, Chip, IconButton, Slider, Checkbox, FormControlLabel, Divider, Snackbar, Alert } from '@mui/material';
import {
  ArrowUpward,
  ArrowDownward,
  ArrowBack,
  ArrowForward,
  Stop,
  KeyboardArrowUp,
  KeyboardArrowDown,
  RotateLeft,
  RotateRight,
  Speed,
  SportsEsports,
} from '@mui/icons-material';

function VehicleControl({ 
  connected, 
  controlEnabled, 
  enableControl, 
  disableControl, 
  publishCmdVel, 
  stopVehicle,
  ros,
  fancyEffects = true
}) {
  const publishIntervalRef = useRef(null);
  const [linearSpeed, setLinearSpeed] = useState(0.2);
  const [angularSpeed, setAngularSpeed] = useState(0.2);
  const [xboxMode, setXboxMode] = useState(false);
  const [deviceId, setDeviceId] = useState(1); // Default to js1
  const [teleopRunning, setTeleopRunning] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const startPublishing = (linear, angular) => {
    // Stop any existing interval
    if (publishIntervalRef.current) {
      clearInterval(publishIntervalRef.current);
    }
    
    // Publish immediately
    publishCmdVel(linear, angular);
    
    // Then publish at 20Hz (every 50ms) like the PyQt5 version
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
      setSnackbar({ 
        open: true, 
        message: 'ROS not connected! Please check connection.', 
        severity: 'error' 
      });
      return;
    }

    // First check if service exists
    ros.getServices((services) => {
      const serviceExists = services.includes('/taluy/web_gui/start_teleop');
      
      if (!serviceExists) {
        setSnackbar({ 
          open: true, 
          message: 'Start teleop service not available. Please restart the web GUI services.', 
          severity: 'error' 
        });
        return;
      }

      try {
        // Set the device_id parameter before calling the service
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

        const request = new window.ROSLIB.ServiceRequest({
          data: xboxMode
        });

        service.callService(request, 
          (result) => {
            if (result.success) {
              setTeleopRunning(true);
              setSnackbar({ 
                open: true, 
                message: `Teleop started with ${xboxMode ? 'Xbox' : 'default'} controller. Make sure your controller is connected!`, 
                severity: 'success' 
              });
            } else {
              setSnackbar({ 
                open: true, 
                message: `Failed to start teleop: ${result.message}`, 
                severity: 'error' 
              });
            }
          },
          (error) => {
            setSnackbar({ 
              open: true, 
              message: `Service call failed: ${error}`, 
              severity: 'error' 
            });
          }
        );
      } catch (error) {
        setSnackbar({ 
          open: true, 
          message: `Error starting teleop: ${error}`, 
          severity: 'error' 
        });
      }
    }, (error) => {
      setSnackbar({ 
        open: true, 
        message: 'Could not check available services', 
        severity: 'error' 
      });
    });
  };

  const handleStopTeleop = async () => {
    if (!ros || !ros.isConnected) {
      setSnackbar({ 
        open: true, 
        message: 'ROS not connected!', 
        severity: 'error' 
      });
      return;
    }

    // First check if service exists
    ros.getServices((services) => {
      const serviceExists = services.includes('/taluy/web_gui/stop_teleop');
      
      if (!serviceExists) {
        setSnackbar({ 
          open: true, 
          message: 'Stop teleop service not available. Please restart the web GUI.', 
          severity: 'warning' 
        });
        // Still set it as not running since the service isn't there
        setTeleopRunning(false);
        return;
      }

      try {
        const service = new window.ROSLIB.Service({
          ros: ros,
          name: '/taluy/web_gui/stop_teleop',
          serviceType: 'std_srvs/Trigger'
        });

        const request = new window.ROSLIB.ServiceRequest({});

        service.callService(request, 
          (result) => {
            if (result.success) {
              setTeleopRunning(false);
              setSnackbar({ 
                open: true, 
                message: 'Teleop stopped successfully', 
                severity: 'success' 
              });
            } else {
              setTeleopRunning(false); // Still mark as stopped even if service reports failure
              setSnackbar({ 
                open: true, 
                message: `Failed to stop teleop: ${result.message}`, 
                severity: 'warning' 
              });
            }
          },
          (error) => {
            setTeleopRunning(false); // Mark as stopped on error
            setSnackbar({ 
              open: true, 
              message: `Service call failed: ${error}`, 
              severity: 'error' 
            });
          }
        );
      } catch (error) {
        setTeleopRunning(false);
        setSnackbar({ 
          open: true, 
          message: `Error stopping teleop: ${error}`, 
          severity: 'error' 
        });
      }
    }, (error) => {
      setSnackbar({ 
        open: true, 
        message: 'Could not check available services', 
        severity: 'error' 
      });
    });
  };

  return (
    <Card elevation={3}>
      <CardContent>
        {/* Header with Status Indicator */}
        <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
          <Typography variant="h6">
            🎮 Vehicle Control
          </Typography>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              px: 1.5,
              py: 0.5,
              borderRadius: 2,
              background: fancyEffects ? (
                controlEnabled 
                  ? 'linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.05) 100%)'
                  : 'linear-gradient(135deg, rgba(158, 158, 158, 0.15) 0%, rgba(158, 158, 158, 0.05) 100%)'
              ) : (
                controlEnabled ? 'rgba(76, 175, 80, 0.1)' : 'rgba(158, 158, 158, 0.1)'
              ),
              border: controlEnabled ? '1px solid rgba(76, 175, 80, 0.3)' : '1px solid rgba(158, 158, 158, 0.2)',
              boxShadow: (fancyEffects && controlEnabled) ? '0 0 12px rgba(76, 175, 80, 0.2)' : 'none',
              transition: fancyEffects ? 'all 0.3s ease' : 'none',
            }}
          >
            <Box
              sx={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                bgcolor: controlEnabled ? '#4caf50' : '#9e9e9e',
                boxShadow: (fancyEffects && controlEnabled) ? '0 0 8px rgba(76, 175, 80, 0.8)' : 'none',
                animation: (fancyEffects && controlEnabled) ? 'pulse 2s ease-in-out infinite' : 'none',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                },
              }}
            />
            <Typography 
              variant="caption" 
              sx={{ 
                fontWeight: 600,
                color: controlEnabled ? '#4caf50' : '#9e9e9e',
                letterSpacing: 0.5,
              }}
            >
              {controlEnabled ? 'ENABLED' : 'DISABLED'}
            </Typography>
          </Box>
        </Box>
        
        {/* Teleoperation Section */}
        <Box 
          sx={{ 
            mb: 2, 
            p: 2, 
            bgcolor: 'rgba(255, 255, 255, 0.03)',
            borderRadius: 2,
            border: '1px solid rgba(255, 255, 255, 0.1)'
          }}
        >
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            <SportsEsports fontSize="small" />
            <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
              Teleoperation (Joystick)
            </Typography>
          </Box>
          
          <Typography variant="caption" sx={{ display: 'block', mb: 1.5, color: 'rgba(255, 255, 255, 0.5)' }}>
            Connect your controller before starting. Using device: <code style={{ color: '#00D9FF' }}>/dev/input/js{deviceId}</code>
          </Typography>
          
          <Box display="flex" gap={1} alignItems="center" flexWrap="wrap">
            <Button
              variant="contained"
              color="primary"
              size="small"
              onClick={handleStartTeleop}
              disabled={!connected || teleopRunning}
              sx={{ minWidth: 100 }}
            >
              Start Teleop
            </Button>
            <Button
              variant="outlined"
              color="error"
              size="small"
              onClick={handleStopTeleop}
              disabled={!connected || !teleopRunning}
              sx={{ minWidth: 100 }}
            >
              Stop Teleop
            </Button>
            <FormControlLabel
              control={
                <Checkbox
                  checked={xboxMode}
                  onChange={(e) => setXboxMode(e.target.checked)}
                  disabled={teleopRunning}
                  size="small"
                  sx={{
                    color: 'rgba(255, 255, 255, 0.5)',
                    '&.Mui-checked': {
                      color: '#00D9FF',
                    },
                  }}
                />
              }
              label={
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Xbox Controller
                </Typography>
              }
            />
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, ml: 1 }}>
              <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                Device:
              </Typography>
              <select
                value={deviceId}
                onChange={(e) => setDeviceId(parseInt(e.target.value))}
                disabled={teleopRunning}
                style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  color: '#fff',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  borderRadius: '4px',
                  padding: '2px 6px',
                  fontSize: '12px',
                  cursor: teleopRunning ? 'not-allowed' : 'pointer',
                }}
              >
                <option value={0}>js0</option>
                <option value={1}>js1</option>
                <option value={2}>js2</option>
                <option value={3}>js3</option>
              </select>
            </Box>
          </Box>
        </Box>

        <Divider sx={{ my: 2, opacity: 0.1 }} />

        {/* Control Enable/Disable Buttons */}
        <Box display="flex" gap={1} mb={2} justifyContent="center">
          <Button
            variant={controlEnabled ? "outlined" : "contained"}
            color="success"
            size="small"
            onClick={enableControl}
            disabled={!connected || controlEnabled}
          >
            Enable Control
          </Button>
          <Button
            variant={controlEnabled ? "contained" : "outlined"}
            color="error"
            size="small"
            onClick={disableControl}
            disabled={!connected || !controlEnabled}
          >
            Disable Control
          </Button>
        </Box>

        <Box display="flex" gap={3} justifyContent="center">
          {/* Left Section: Forward/Back/Left/Right */}
          <Box display="flex" flexDirection="column" alignItems="center" gap={1}>
            <Typography variant="caption" color="text.secondary" mb={0.5}>
              Translation
            </Typography>
            {/* Forward */}
            <IconButton
              color="primary"
              size="large"
              disabled={!connected}
              sx={{ 
                bgcolor: 'primary.dark', 
                '&:hover': { bgcolor: 'primary.main' },
                width: 60,
                height: 60,
              }}
              onMouseDown={() => startPublishing({ x: linearSpeed, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })}
              onMouseUp={stopPublishing}
              onMouseLeave={stopPublishing}
              onTouchStart={() => startPublishing({ x: linearSpeed, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })}
              onTouchEnd={stopPublishing}
            >
              <ArrowUpward fontSize="large" />
            </IconButton>

            {/* Left, Stop, Right */}
            <Box display="flex" gap={1}>
              <IconButton
                color="primary"
                size="large"
                disabled={!connected}
                sx={{ bgcolor: 'primary.dark', width: 60, height: 60 }}
                onMouseDown={() => startPublishing({ x: 0, y: linearSpeed, z: 0 }, { x: 0, y: 0, z: 0 })}
                onMouseUp={stopPublishing}
                onMouseLeave={stopPublishing}
                onTouchStart={() => startPublishing({ x: 0, y: linearSpeed, z: 0 }, { x: 0, y: 0, z: 0 })}
                onTouchEnd={stopPublishing}
              >
                <ArrowBack fontSize="large" />
              </IconButton>

              <IconButton
                color="error"
                size="large"
                disabled={!connected}
                sx={{ bgcolor: 'error.dark', width: 60, height: 60 }}
                onClick={stopPublishing}
              >
                <Stop fontSize="large" />
              </IconButton>

              <IconButton
                color="primary"
                size="large"
                disabled={!connected}
                sx={{ bgcolor: 'primary.dark', width: 60, height: 60 }}
                onMouseDown={() => startPublishing({ x: 0, y: -linearSpeed, z: 0 }, { x: 0, y: 0, z: 0 })}
                onMouseUp={stopPublishing}
                onMouseLeave={stopPublishing}
                onTouchStart={() => startPublishing({ x: 0, y: -linearSpeed, z: 0 }, { x: 0, y: 0, z: 0 })}
                onTouchEnd={stopPublishing}
              >
                <ArrowForward fontSize="large" />
              </IconButton>
            </Box>

            {/* Backward */}
            <IconButton
              color="primary"
              size="large"
              disabled={!connected}
              sx={{ bgcolor: 'primary.dark', width: 60, height: 60 }}
              onMouseDown={() => startPublishing({ x: -linearSpeed, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })}
              onMouseUp={stopPublishing}
              onMouseLeave={stopPublishing}
              onTouchStart={() => startPublishing({ x: -linearSpeed, y: 0, z: 0 }, { x: 0, y: 0, z: 0 })}
              onTouchEnd={stopPublishing}
            >
              <ArrowDownward fontSize="large" />
            </IconButton>

            {/* Linear Speed Slider */}
            <Box sx={{ width: 180, mt: 2, px: 2 }}>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Speed fontSize="small" color="primary" />
                <Typography variant="caption" color="text.secondary">
                  Linear Speed
                </Typography>
              </Box>
              <Slider
                value={linearSpeed}
                onChange={(e, value) => setLinearSpeed(value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks={[
                  { value: 0.1, label: '0.1' },
                  { value: 1.0, label: '1.0' },
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => value.toFixed(2)}
                disabled={!connected}
                sx={fancyEffects ? {
                  '& .MuiSlider-thumb': {
                    boxShadow: '0 0 8px rgba(0, 217, 255, 0.6)',
                  },
                  '& .MuiSlider-track': {
                    background: 'linear-gradient(90deg, #00D9FF 0%, #7C4DFF 100%)',
                  },
                } : {}}
              />
            </Box>
          </Box>

          {/* Right Section: Up/Down (Z) and Yaw (Angular) */}
          <Box display="flex" flexDirection="column" alignItems="center" gap={1}>
            <Typography variant="caption" color="text.secondary" mb={0.5}>
              Depth & Rotation
            </Typography>
            {/* Up (Z positive) */}
            <IconButton
              color="secondary"
              size="large"
              disabled={!connected}
              sx={{ 
                bgcolor: 'secondary.dark', 
                '&:hover': { bgcolor: 'secondary.main' },
                width: 60,
                height: 60,
              }}
              onMouseDown={() => startPublishing({ x: 0, y: 0, z: linearSpeed }, { x: 0, y: 0, z: 0 })}
              onMouseUp={stopPublishing}
              onMouseLeave={stopPublishing}
              onTouchStart={() => startPublishing({ x: 0, y: 0, z: linearSpeed }, { x: 0, y: 0, z: 0 })}
              onTouchEnd={stopPublishing}
            >
              <KeyboardArrowUp fontSize="large" />
            </IconButton>

            {/* Yaw Left (Angular Z positive) and Yaw Right (Angular Z negative) */}
            <Box display="flex" gap={1}>
              <IconButton
                color="secondary"
                size="large"
                disabled={!connected}
                sx={{ bgcolor: 'secondary.dark', width: 60, height: 60 }}
                onMouseDown={() => startPublishing({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: angularSpeed })}
                onMouseUp={stopPublishing}
                onMouseLeave={stopPublishing}
                onTouchStart={() => startPublishing({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: angularSpeed })}
                onTouchEnd={stopPublishing}
              >
                <RotateLeft fontSize="large" />
              </IconButton>

              <IconButton
                color="secondary"
                size="large"
                disabled={!connected}
                sx={{ bgcolor: 'secondary.dark', width: 60, height: 60 }}
                onMouseDown={() => startPublishing({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -angularSpeed })}
                onMouseUp={stopPublishing}
                onMouseLeave={stopPublishing}
                onTouchStart={() => startPublishing({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -angularSpeed })}
                onTouchEnd={stopPublishing}
              >
                <RotateRight fontSize="large" />
              </IconButton>
            </Box>

            {/* Down (Z negative) */}
            <IconButton
              color="secondary"
              size="large"
              disabled={!connected}
              sx={{ bgcolor: 'secondary.dark', width: 60, height: 60 }}
              onMouseDown={() => startPublishing({ x: 0, y: 0, z: -linearSpeed }, { x: 0, y: 0, z: 0 })}
              onMouseUp={stopPublishing}
              onMouseLeave={stopPublishing}
              onTouchStart={() => startPublishing({ x: 0, y: 0, z: -linearSpeed }, { x: 0, y: 0, z: 0 })}
              onTouchEnd={stopPublishing}
            >
              <KeyboardArrowDown fontSize="large" />
            </IconButton>

            {/* Angular Speed Slider */}
            <Box sx={{ width: 180, mt: 2, px: 2 }}>
              <Box display="flex" alignItems="center" gap={1} mb={1}>
                <Speed fontSize="small" color="secondary" />
                <Typography variant="caption" color="text.secondary">
                  Angular Speed
                </Typography>
              </Box>
              <Slider
                value={angularSpeed}
                onChange={(e, value) => setAngularSpeed(value)}
                min={0.1}
                max={1.0}
                step={0.05}
                marks={[
                  { value: 0.1, label: '0.1' },
                  { value: 1.0, label: '1.0' },
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => value.toFixed(2)}
                disabled={!connected}
                sx={fancyEffects ? {
                  '& .MuiSlider-thumb': {
                    boxShadow: '0 0 8px rgba(124, 77, 255, 0.6)',
                  },
                  '& .MuiSlider-track': {
                    background: 'linear-gradient(90deg, #7C4DFF 0%, #00D9FF 100%)',
                  },
                } : {}}
              />
            </Box>
          </Box>
        </Box>
      </CardContent>

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setSnackbar({ ...snackbar, open: false })} 
          severity={snackbar.severity}
          sx={{ width: '100%' }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Card>
  );
}

export default VehicleControl;

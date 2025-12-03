import React, { useState } from 'react';
import { Card, CardContent, Typography, Button, Grid, Switch, FormControlLabel, Snackbar, Alert } from '@mui/material';
import { PlayArrow, StopCircle, CameraAlt } from '@mui/icons-material';

function ObjectDetection({ 
  connected,
  ros,
  cudaEnabled, 
  setCudaEnabled, 
  detectionRunning, 
  setDetectionRunning 
}) {
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });

  const handleStartDetection = async () => {
    if (!connected || !ros) {
      setSnackbar({ open: true, message: 'Not connected to ROS', severity: 'error' });
      return;
    }

    try {
      const service = new window.ROSLIB.Service({
        ros: ros,
        name: '/taluy/web_gui/start_detection',
        serviceType: 'std_srvs/SetBool'
      });

      const request = new window.ROSLIB.ServiceRequest({
        data: cudaEnabled  // true for CUDA, false for CPU
      });

      service.callService(request, (result) => {
        if (result.success) {
          setDetectionRunning(true);
          const device = cudaEnabled ? 'CUDA (GPU)' : 'CPU';
          setSnackbar({ open: true, message: `Detection started with ${device}`, severity: 'success' });
        } else {
          setSnackbar({ open: true, message: `Failed: ${result.message}`, severity: 'error' });
        }
      }, (error) => {
        console.error('Service call failed:', error);
        setSnackbar({ 
          open: true, 
          message: 'Detection service not available. Make sure web_gui.launch is running.', 
          severity: 'error' 
        });
      });
    } catch (error) {
      console.error('Error calling service:', error);
      setSnackbar({ open: true, message: 'Failed to call service', severity: 'error' });
    }
  };

  const handleStopDetection = async () => {
    if (!connected || !ros) {
      setSnackbar({ open: true, message: 'Not connected to ROS', severity: 'error' });
      return;
    }

    try {
      const service = new window.ROSLIB.Service({
        ros: ros,
        name: '/taluy/web_gui/stop_detection',
        serviceType: 'std_srvs/Trigger'
      });

      const request = new window.ROSLIB.ServiceRequest({});

      service.callService(request, (result) => {
        if (result.success) {
          setDetectionRunning(false);
          setSnackbar({ open: true, message: 'Detection stopped', severity: 'success' });
        } else {
          setSnackbar({ open: true, message: `Failed: ${result.message}`, severity: 'error' });
        }
      }, (error) => {
        console.error('Service call failed:', error);
        setSnackbar({ 
          open: true, 
          message: 'Detection service not available.', 
          severity: 'error' 
        });
      });
    } catch (error) {
      console.error('Error calling service:', error);
      setSnackbar({ open: true, message: 'Failed to call service', severity: 'error' });
    }
  };

  const handleLaunchRqt = async () => {
    if (!connected || !ros) {
      setSnackbar({ open: true, message: 'Not connected to ROS', severity: 'error' });
      return;
    }

    try {
      const service = new window.ROSLIB.Service({
        ros: ros,
        name: '/taluy/web_gui/launch_rqt_image_view',
        serviceType: 'std_srvs/Trigger'
      });

      const request = new window.ROSLIB.ServiceRequest({});

      service.callService(request, (result) => {
        if (result.success) {
          setSnackbar({ open: true, message: 'rqt_image_view launched successfully!', severity: 'success' });
        } else {
          setSnackbar({ open: true, message: `Failed: ${result.message}`, severity: 'error' });
        }
      }, (error) => {
        console.error('Service call failed:', error);
        setSnackbar({ 
          open: true, 
          message: 'Service not available. Make sure web_gui.launch is running.', 
          severity: 'error' 
        });
      });
    } catch (error) {
      console.error('Error calling service:', error);
      setSnackbar({ open: true, message: 'Failed to call service', severity: 'error' });
    }
  };
  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" mb={2}>🎮 Object Detection</Typography>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12}>
            <FormControlLabel
              control={
                <Switch
                  checked={cudaEnabled}
                  onChange={(e) => setCudaEnabled(e.target.checked)}
                  disabled={!connected || detectionRunning}
                />
              }
              label="Use CUDA (GPU Acceleration)"
            />
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color="success"
              startIcon={<PlayArrow />}
              onClick={handleStartDetection}
              disabled={!connected || detectionRunning}
            >
              Start Detection
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color="error"
              startIcon={<StopCircle />}
              onClick={handleStopDetection}
              disabled={!connected || !detectionRunning}
            >
              Stop Detection
            </Button>
          </Grid>
          <Grid item xs={12}>
            <Button
              fullWidth
              variant="outlined"
              startIcon={<CameraAlt />}
              disabled={!connected}
              onClick={handleLaunchRqt}
            >
              Open rqt_image_view
            </Button>
          </Grid>
        </Grid>

        <Snackbar
          open={snackbar.open}
          autoHideDuration={4000}
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
      </CardContent>
    </Card>
  );
}

export default ObjectDetection;

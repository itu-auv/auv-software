import React, { useState } from 'react';
import { Play, StopCircle, Camera, Eye, Cpu } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';

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
        data: cudaEnabled
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
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <Eye className="w-4 h-4 text-white/50" />
            Object Detection
          </span>
          <span className={`text-xs font-medium px-2 py-1 rounded-full ${
            detectionRunning
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-white/5 text-white/30 border border-white/10'
          }`}>
            {detectionRunning ? 'Running' : 'Stopped'}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between p-3 rounded-xl border border-white/[0.06] bg-white/[0.02]">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-white/40" />
            <span className="text-sm text-white/70">CUDA Acceleration</span>
          </div>
          <Switch
            checked={cudaEnabled}
            onCheckedChange={setCudaEnabled}
            disabled={!connected || detectionRunning}
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="success"
            onClick={handleStartDetection}
            disabled={!connected || detectionRunning}
          >
            <Play className="w-4 h-4 mr-2" />
            Start
          </Button>
          <Button
            variant="secondary"
            onClick={handleStopDetection}
            disabled={!connected || !detectionRunning}
          >
            <StopCircle className="w-4 h-4 mr-2" />
            Stop
          </Button>
        </div>

        <Button
          variant="outline"
          className="w-full"
          disabled={!connected}
          onClick={handleLaunchRqt}
        >
          <Camera className="w-4 h-4 mr-2" />
          Open rqt_image_view
        </Button>
      </CardContent>
    </Card>
  );
}

export default ObjectDetection;

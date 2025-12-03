import React from 'react';
import { Box, Card, CardContent, Typography, Grid, Chip, Alert } from '@mui/material';
import { Battery90 } from '@mui/icons-material';

function BatteryStatus({ 
  voltage, 
  current, 
  power, 
  powerHistory, 
  powerTopicName, 
  lastPowerUpdate, 
  powerSource 
}) {
  const getBatteryColor = () => {
    if (voltage === 0) return 'default';
    if (voltage < 14.8) return 'error';
    if (voltage < 15.2) return 'warning';
    return 'success';
  };

  const getBatteryLabel = () => {
    if (voltage === 0) return 'NO DATA';
    if (voltage < 14.8) return 'CRITICAL';
    if (voltage < 15.2) return 'LOW';
    return 'GOOD';
  };

  const getPowerSourceDisplay = () => {
    if (powerSource === 'simulation') return '🎮 Simulation (Fixed 16V)';
    if (powerSource === 'hardware') return '🔌 Real Hardware';
    return '❓ Unknown';
  };

  return (
    <Card elevation={3}>
      <CardContent>
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Battery90 color={getBatteryColor()} fontSize="large" />
          <Box flexGrow={1}>
            <Typography variant="h6">Battery Status</Typography>
            <Typography variant="caption" sx={{ opacity: 0.7 }}>
              {getPowerSourceDisplay()}
            </Typography>
          </Box>
        </Box>
        
        {voltage > 0 ? (
          <>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Typography variant="caption" sx={{ opacity: 0.7 }}>Voltage</Typography>
                <Typography variant="h4" color={`${getBatteryColor()}.main`} fontWeight="bold">
                  {voltage.toFixed(2)}V
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" sx={{ opacity: 0.7 }}>Current</Typography>
                <Typography variant="h4" fontWeight="bold">
                  {current.toFixed(2)}A
                </Typography>
              </Grid>
              <Grid item xs={12}>
                <Typography variant="caption" sx={{ opacity: 0.7 }}>Power Consumption</Typography>
                <Typography variant="h5" fontWeight="bold" color="warning.main">
                  {power.toFixed(2)}W
                </Typography>
              </Grid>
            </Grid>
            
            <Chip
              label={getBatteryLabel()}
              color={getBatteryColor()}
              sx={{ mt: 2 }}
            />
            
            {/* Power History Graph */}
            {powerHistory.length > 1 && (
              <Box sx={{ mt: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                <Typography variant="caption" sx={{ opacity: 0.7 }}>Power History (60s)</Typography>
                <Box sx={{ position: 'relative', height: 60, mt: 1 }}>
                  <svg width="100%" height="60" style={{ display: 'block' }}>
                    <polyline
                      fill="none"
                      stroke="#FF9800"
                      strokeWidth="2"
                      points={powerHistory.map((point, index) => {
                        const x = (index / (powerHistory.length - 1)) * 100;
                        const maxPower = Math.max(...powerHistory.map(p => p.power), 1);
                        const y = 60 - (point.power / maxPower) * 50;
                        return `${x}%,${y}`;
                      }).join(' ')}
                    />
                  </svg>
                  <Typography variant="caption" sx={{ position: 'absolute', top: 0, right: 0, opacity: 0.5 }}>
                    {Math.max(...powerHistory.map(p => p.power)).toFixed(0)}W
                  </Typography>
                  <Typography variant="caption" sx={{ position: 'absolute', bottom: 0, right: 0, opacity: 0.5 }}>
                    0W
                  </Typography>
                </Box>
              </Box>
            )}
            
            {/* Debug Info */}
            <Box sx={{ mt: 2, p: 1, bgcolor: 'background.paper', borderRadius: 1, fontSize: '0.7rem' }}>
              <Typography variant="caption" display="block">
                <strong>Topic:</strong> {powerTopicName}
              </Typography>
              <Typography variant="caption" display="block">
                <strong>Last Update:</strong> {lastPowerUpdate ? lastPowerUpdate.toLocaleTimeString() : 'Never'}
              </Typography>
              {powerSource === 'simulation' && (
                <Typography variant="caption" display="block" sx={{ color: 'info.main', mt: 0.5 }}>
                  ℹ️ Simulated voltage (fixed 16V in Gazebo)
                </Typography>
              )}
            </Box>
          </>
        ) : (
          <Alert severity={powerTopicName === 'NOT_FOUND' ? 'error' : 'info'} sx={{ mt: 1 }}>
            {powerTopicName === 'NOT_FOUND' ? (
              <strong>No power topic found! Make sure simulation or power node is running.</strong>
            ) : (
              <>
                Waiting for battery data...
                <Typography variant="caption" display="block" sx={{ mt: 1 }}>
                  Topic: <code>{powerTopicName || 'Searching...'}</code>
                </Typography>
              </>
            )}
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}

export default BatteryStatus;

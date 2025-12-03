import React, { useState } from 'react';
import { Card, CardContent, Typography, Button, Grid, Box, Switch } from '@mui/material';
import { LocationOn, Delete, Refresh, Sensors } from '@mui/icons-material';

function ServicesPanel({ 
  connected, 
  startLocalization,
  enableDVL, 
  disableDVL, 
  clearObjects, 
  resetPose 
}) {
  const [dvlEnabled, setDvlEnabled] = useState(false);

  const handleDvlToggle = (event) => {
    const enabled = event.target.checked;
    setDvlEnabled(enabled);
    if (enabled) {
      enableDVL();
    } else {
      disableDVL();
    }
  };

  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" mb={2}>⚙️ Services</Typography>
        <Grid container spacing={2}>
          {/* Localization Button */}
          <Grid item xs={12}>
            <Button
              fullWidth
              variant="contained"
              startIcon={<LocationOn />}
              onClick={startLocalization}
              disabled={!connected}
              sx={{
                py: 1.5,
                background: 'linear-gradient(135deg, #00D9FF 0%, #7C4DFF 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #00B8D4 0%, #651FFF 100%)',
                },
                '&:disabled': {
                  background: 'rgba(255, 255, 255, 0.12)',
                },
              }}
            >
              Start Localization
            </Button>
          </Grid>

          {/* DVL Toggle */}
          <Grid item xs={12}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 2,
                bgcolor: dvlEnabled 
                  ? 'rgba(0, 224, 150, 0.1)' 
                  : 'rgba(255, 255, 255, 0.05)',
                borderRadius: 2,
                border: `1px solid ${dvlEnabled ? 'rgba(0, 224, 150, 0.5)' : 'rgba(255, 255, 255, 0.1)'}`,
                transition: 'all 0.3s ease',
                '&:hover': {
                  bgcolor: dvlEnabled 
                    ? 'rgba(0, 224, 150, 0.15)' 
                    : 'rgba(255, 255, 255, 0.08)',
                },
              }}
            >
              <Box display="flex" alignItems="center" gap={1.5}>
                <Sensors 
                  sx={{ 
                    color: dvlEnabled ? '#00E096' : 'rgba(255, 255, 255, 0.5)',
                    fontSize: 28,
                  }} 
                />
                <Box>
                  <Typography variant="body1" sx={{ fontWeight: 600, color: '#fff' }}>
                    DVL Sensor
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                    {dvlEnabled ? 'Enabled' : 'Disabled'}
                  </Typography>
                </Box>
              </Box>
              <Switch
                checked={dvlEnabled}
                onChange={handleDvlToggle}
                disabled={!connected}
                sx={{
                  '& .MuiSwitch-switchBase.Mui-checked': {
                    color: '#00E096',
                  },
                  '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                    backgroundColor: '#00E096',
                  },
                }}
              />
            </Box>
          </Grid>

          {/* Action Buttons */}
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              color="warning"
              startIcon={<Delete />}
              onClick={clearObjects}
              disabled={!connected}
              sx={{
                borderWidth: 2,
                '&:hover': {
                  borderWidth: 2,
                },
              }}
            >
              Clear Objects
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="outlined"
              color="secondary"
              startIcon={<Refresh />}
              onClick={resetPose}
              disabled={!connected}
              sx={{
                borderWidth: 2,
                '&:hover': {
                  borderWidth: 2,
                },
              }}
            >
              Reset Pose
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

export default ServicesPanel;

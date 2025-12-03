import React from 'react';
import { Card, CardContent, Typography, Button, Grid } from '@mui/material';

function MissionControls({ 
  connected, 
  launchTorpedo1, 
  launchTorpedo2, 
  dropBall 
}) {
  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" mb={2}>🎯 Mission Controls</Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color="error"
              onClick={launchTorpedo1}
              disabled={!connected}
              sx={{ fontWeight: 'bold' }}
            >
              🚀 Launch Torpedo 1
            </Button>
          </Grid>
          <Grid item xs={6}>
            <Button
              fullWidth
              variant="contained"
              color="error"
              onClick={launchTorpedo2}
              disabled={!connected}
              sx={{ fontWeight: 'bold' }}
            >
              🚀 Launch Torpedo 2
            </Button>
          </Grid>
          <Grid item xs={12}>
            <Button
              fullWidth
              variant="contained"
              color="warning"
              onClick={dropBall}
              disabled={!connected}
              sx={{ fontWeight: 'bold' }}
            >
              ⚫ Drop Ball
            </Button>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
}

export default MissionControls;

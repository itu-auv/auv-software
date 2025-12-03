import React from 'react';
import { Box, Card, CardContent, Typography, Button, Grid, Switch, FormControlLabel } from '@mui/material';
import { PlayArrow, StopCircle } from '@mui/icons-material';

function StateMachine({ 
  connected, 
  testMode, 
  setTestMode, 
  smachRunning, 
  setSmachRunning, 
  selectedStates, 
  setSelectedStates 
}) {
  return (
    <Card elevation={3}>
      <CardContent>
        <Typography variant="h6" mb={2}>🤖 State Machine (SMACH)</Typography>
        
        <Grid container spacing={2} mb={2}>
          <Grid item>
            <FormControlLabel
              control={
                <Switch
                  checked={testMode}
                  onChange={(e) => setTestMode(e.target.checked)}
                  disabled={!connected || smachRunning}
                />
              }
              label="Test Mode"
            />
          </Grid>
          <Grid item>
            <Button
              variant="contained"
              color="success"
              startIcon={<PlayArrow />}
              onClick={() => setSmachRunning(true)}
              disabled={!connected || smachRunning}
            >
              Launch SMACH
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="contained"
              color="error"
              startIcon={<StopCircle />}
              onClick={() => setSmachRunning(false)}
              disabled={!connected || !smachRunning}
            >
              Stop SMACH
            </Button>
          </Grid>
        </Grid>

        {testMode && (
          <Box>
            <Typography variant="subtitle2" mb={1}>Select Test States:</Typography>
            <Grid container spacing={1}>
              {Object.keys(selectedStates).map((state) => (
                <Grid item key={state}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={selectedStates[state]}
                        onChange={(e) => setSelectedStates(prev => ({
                          ...prev,
                          [state]: e.target.checked
                        }))}
                        disabled={!connected || smachRunning}
                        size="small"
                      />
                    }
                    label={state.charAt(0).toUpperCase() + state.slice(1)}
                  />
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

export default StateMachine;

import React from 'react';
import { Box, Card, CardContent, Typography, Button, Slider } from '@mui/material';
import { Waves } from '@mui/icons-material';

function DepthControl({ depth, setDepth, setDepthService, connected }) {
  return (
    <Card elevation={3}>
      <CardContent>
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Waves color="primary" fontSize="large" />
          <Typography variant="h6">Depth Control</Typography>
        </Box>
        <Slider
          value={depth}
          onChange={(e, value) => setDepth(value)}
          min={-3.0}
          max={0.0}
          step={0.1}
          marks={[
            { value: -3.0, label: '-3m' },
            { value: -1.5, label: '-1.5m' },
            { value: 0, label: '0m' },
          ]}
          valueLabelDisplay="on"
          sx={{ mt: 2, mb: 2 }}
        />
        <Button
          variant="contained"
          fullWidth
          onClick={setDepthService}
          disabled={!connected}
        >
          Set Depth
        </Button>
      </CardContent>
    </Card>
  );
}

export default DepthControl;

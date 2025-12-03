import React from 'react';
import { Box, AppBar, Toolbar, Typography, Button, Chip, CircularProgress } from '@mui/material';
import { Refresh } from '@mui/icons-material';

function Header({ connected, connecting, connectToROS, fancyEffects = true }) {
  return (
    <AppBar position="sticky" elevation={2} sx={{ width: '100%' }}>
      <Toolbar sx={{ width: '100%', maxWidth: '100%', px: 3 }}>
        <Box display="flex" alignItems="center" gap={2} sx={{ flexGrow: 1 }}>
          <img 
            src="/logo.png" 
            alt="ITU AUV Logo" 
            style={{ height: 40, width: 'auto' }}
          />
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            <Typography 
              variant="h5" 
              sx={fancyEffects ? (theme) => ({ 
                fontWeight: 800,
                letterSpacing: '0.5px',
                color: theme.palette.primary.main,
                animation: 'glow-pulse 2s ease-in-out infinite',
                '@keyframes glow-pulse': {
                  '0%, 100%': { 
                    textShadow: `0 0 10px ${theme.palette.primary.main}80, 0 0 20px ${theme.palette.primary.main}40`,
                  },
                  '50%': { 
                    textShadow: `0 0 20px ${theme.palette.primary.main}, 0 0 30px ${theme.palette.primary.main}80, 0 0 40px ${theme.palette.primary.main}40`,
                  },
                },
              }) : (theme) => ({
                fontWeight: 800,
                letterSpacing: '0.5px',
                color: theme.palette.primary.main,
              })}
            >
              ITU AUV CONTROL PANEL
            </Typography>
            <Typography 
              variant="caption" 
              sx={(theme) => ({ 
                color: theme.palette.primary.main,
                opacity: 0.7,
                fontWeight: 500,
                letterSpacing: '2px',
                fontSize: '0.65rem',
                textTransform: 'uppercase',
              })}
            >
              Autonomous Underwater Vehicle System
            </Typography>
          </Box>
        </Box>
        <Box display="flex" alignItems="center" gap={2}>
          {connecting && <CircularProgress size={20} />}
          <Chip
            icon={
              <Box sx={{ 
                width: 10, 
                height: 10, 
                borderRadius: '50%', 
                bgcolor: connected ? 'success.main' : 'error.main',
                animation: (connected && fancyEffects) ? 'pulse 2s infinite' : 'none',
                '@keyframes pulse': {
                  '0%, 100%': { opacity: 1 },
                  '50%': { opacity: 0.5 },
                }
              }} />
            }
            label={connected ? 'Connected' : 'Disconnected'}
            color={connected ? 'success' : 'error'}
            variant="outlined"
          />
          {!connected && (
            <Button
              variant="outlined"
              size="small"
              startIcon={<Refresh />}
              onClick={connectToROS}
            >
              Reconnect
            </Button>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;

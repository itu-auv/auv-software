import React from 'react';
import { Box, Card, CardContent, Typography, Button } from '@mui/material';
import { Pool, Computer } from '@mui/icons-material';
import VehicleModel3DBackground from './VehicleModel3DBackground';

function StartScreen({ onSelectMode }) {
  const [modelError, setModelError] = React.useState(false);

  React.useEffect(() => {
    console.log('StartScreen mounted, attempting to load 3D model...');
  }, []);

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
        position: 'relative',
        overflow: 'hidden',
        padding: 4,
      }}
    >
      {/* 3D Model Background */}
      {!modelError && (
        <React.Suspense 
          fallback={
            <Box sx={{ position: 'absolute', top: 10, left: 10, color: 'cyan', zIndex: 999 }}>
              Loading 3D Model...
            </Box>
          }
        >
          <VehicleModel3DBackground color="#00D9FF" />
        </React.Suspense>
      )}

      {/* Background animated elements */}
      <Box
        sx={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          opacity: 0.05,
          background: 'radial-gradient(circle at 20% 50%, #00D9FF 0%, transparent 50%), radial-gradient(circle at 80% 50%, #7C4DFF 0%, transparent 50%)',
          animation: 'pulse 4s ease-in-out infinite',
          zIndex: 1,
          '@keyframes pulse': {
            '0%, 100%': { opacity: 0.05 },
            '50%': { opacity: 0.1 },
          },
        }}
      />

      <Box sx={{ maxWidth: '1200px', width: '100%', position: 'relative', zIndex: 2, display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Title - positioned at top */}
        <Box textAlign="center" sx={{ position: 'absolute', top: '-20vh', left: 0, right: 0, zIndex: 1 }}>
          <Typography 
            variant="h2" 
            sx={{ 
              fontWeight: 700,
              mb: 2,
              background: 'linear-gradient(135deg, #00D9FF 0%, #7C4DFF 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              textShadow: '0 0 40px rgba(0, 217, 255, 0.3)',
            }}
          >
            ITU AUV Control Panel
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
            Select Operation Mode
          </Typography>
        </Box>

        {/* Mode Selection Cards */}
        <Box 
          sx={{ 
            display: 'grid', 
            gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
            gap: 4,
            position: 'relative',
            zIndex: 1,
          }}
        >
          {/* Pool Test Mode */}
          <Card
            sx={{
              background: 'rgba(0, 0, 0, 0.4)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              transition: 'all 0.3s ease',
              cursor: 'pointer',
              '&:hover': {
                transform: 'translateY(-8px)',
                border: '1px solid rgba(0, 217, 255, 0.6)',
                boxShadow: '0 8px 32px rgba(0, 217, 255, 0.4)',
                background: 'rgba(0, 217, 255, 0.15)',
              },
            }}
            onClick={() => onSelectMode('pool')}
          >
            <CardContent sx={{ p: 4, textAlign: 'center' }}>
              <Box
                sx={{
                  width: 80,
                  height: 80,
                  margin: '0 auto 24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'linear-gradient(135deg, #00D9FF 0%, #00A0CC 100%)',
                  borderRadius: '50%',
                  boxShadow: '0 4px 20px rgba(0, 217, 255, 0.4)',
                }}
              >
                <Pool sx={{ fontSize: 40, color: '#fff' }} />
              </Box>
              
              <Typography 
                variant="h4" 
                sx={{ 
                  fontWeight: 600, 
                  mb: 2,
                  color: '#fff',
                }}
              >
                Pool Test
              </Typography>
              
              <Typography 
                variant="body1" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.7)',
                }}
              >
                Real hardware testing in pool environment
              </Typography>

              <Typography 
                variant="caption" 
                sx={{ 
                  display: 'block',
                  mt: 2,
                  color: 'rgba(255, 255, 255, 0.5)',
                  fontStyle: 'italic',
                }}
              >
                Coming soon...
              </Typography>
            </CardContent>
          </Card>

          {/* Simulation Mode */}
          <Card
            sx={{
              background: 'rgba(0, 0, 0, 0.4)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              transition: 'all 0.3s ease',
              cursor: 'pointer',
              '&:hover': {
                transform: 'translateY(-8px)',
                border: '1px solid rgba(124, 77, 255, 0.6)',
                boxShadow: '0 8px 32px rgba(124, 77, 255, 0.4)',
                background: 'rgba(124, 77, 255, 0.15)',
              },
            }}
            onClick={() => onSelectMode('simulation')}
          >
            <CardContent sx={{ p: 4, textAlign: 'center' }}>
              <Box
                sx={{
                  width: 80,
                  height: 80,
                  margin: '0 auto 24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'linear-gradient(135deg, #7C4DFF 0%, #651FFF 100%)',
                  borderRadius: '50%',
                  boxShadow: '0 4px 20px rgba(124, 77, 255, 0.4)',
                }}
              >
                <Computer sx={{ fontSize: 40, color: '#fff' }} />
              </Box>
              
              <Typography 
                variant="h4" 
                sx={{ 
                  fontWeight: 600, 
                  mb: 2,
                  color: '#fff',
                }}
              >
                Simulation
              </Typography>
              
              <Typography 
                variant="body1" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.7)',
                }}
              >
                Virtual testing with Gazebo simulation
              </Typography>
            </CardContent>
          </Card>
        </Box>

        {/* Footer Info */}
        <Box textAlign="center" mt={6} sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
            Istanbul Technical University - Autonomous Underwater Vehicle Team
          </Typography>
        </Box>
      </Box>
    </Box>
  );
}

export default StartScreen;

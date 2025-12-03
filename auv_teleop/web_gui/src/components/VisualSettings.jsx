import React, { useState } from 'react';
import {
  Drawer,
  IconButton,
  Box,
  Typography,
  Divider,
  FormControlLabel,
  Checkbox,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import VisibilityIcon from '@mui/icons-material/Visibility';
import TuneIcon from '@mui/icons-material/Tune';
import InfoIcon from '@mui/icons-material/Info';

const VisualSettings = ({ 
  threeEnabled, 
  setThreeEnabled, 
  currentTheme, 
  setTheme,
  fancyEffects,
  setFancyEffects 
}) => {
  const [open, setOpen] = useState(false);
  const [expandedCategory, setExpandedCategory] = useState('visual');
  const [performanceMetrics, setPerformanceMetrics] = useState({
    fps: 0,
    memory: 0,
    renderTime: 0,
  });

  // Performance monitoring
  React.useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId;

    const measurePerformance = () => {
      frameCount++;
      const currentTime = performance.now();
      const elapsed = currentTime - lastTime;

      if (elapsed >= 1000) {
        const fps = Math.round((frameCount * 1000) / elapsed);
        const memory = performance.memory 
          ? Math.round(performance.memory.usedJSHeapSize / 1048576) 
          : 0;

        setPerformanceMetrics({
          fps: fps,
          memory: memory,
          renderTime: Math.round(elapsed / frameCount * 100) / 100,
        });

        frameCount = 0;
        lastTime = currentTime;
      }

      animationId = requestAnimationFrame(measurePerformance);
    };

    animationId = requestAnimationFrame(measurePerformance);

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, []);

  const handleDrawerOpen = () => {
    setOpen(true);
  };

  const handleDrawerClose = () => {
    setOpen(false);
  };

  const handleCategoryChange = (panel) => (event, isExpanded) => {
    setExpandedCategory(isExpanded ? panel : false);
  };

  const handleThemeChange = (event) => {
    setTheme(event.target.value);
  };

  const handleThreeToggle = (event) => {
    setThreeEnabled(event.target.checked);
  };

  const handleFancyEffectsToggle = (event) => {
    setFancyEffects(event.target.checked);
  };

  return (
    <>
      {/* Floating Settings Button */}
      <Tooltip title="Visual Settings" placement="left">
        <IconButton
          onClick={handleDrawerOpen}
          sx={{
            position: 'fixed',
            right: 16,
            top: '50%',
            transform: 'translateY(-50%)',
            zIndex: 1300,
            bgcolor: 'rgba(0, 217, 255, 0.1)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(0, 217, 255, 0.3)',
            '&:hover': {
              bgcolor: 'rgba(0, 217, 255, 0.2)',
              borderColor: 'rgba(0, 217, 255, 0.5)',
            },
          }}
        >
          <SettingsIcon sx={{ color: '#00D9FF' }} />
        </IconButton>
      </Tooltip>

      {/* Settings Drawer */}
      <Drawer
        anchor="right"
        open={open}
        onClose={handleDrawerClose}
        sx={{
          '& .MuiDrawer-paper': {
            width: 340,
            bgcolor: 'rgba(10, 10, 30, 0.95)',
            backdropFilter: 'blur(20px)',
            borderLeft: '1px solid rgba(0, 217, 255, 0.3)',
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          {/* Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6" sx={{ color: '#00D9FF', fontWeight: 600 }}>
              Settings
            </Typography>
            <IconButton onClick={handleDrawerClose} size="small">
              <ChevronRightIcon sx={{ color: '#00D9FF' }} />
            </IconButton>
          </Box>

          <Divider sx={{ borderColor: 'rgba(0, 217, 255, 0.2)', mb: 2 }} />

          {/* Visual Settings Category */}
          <Accordion
            expanded={expandedCategory === 'visual'}
            onChange={handleCategoryChange('visual')}
            sx={{
              bgcolor: 'rgba(0, 217, 255, 0.05)',
              borderRadius: 1,
              mb: 1,
              '&:before': { display: 'none' },
              border: '1px solid rgba(0, 217, 255, 0.2)',
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon sx={{ color: '#00D9FF' }} />}
              sx={{
                '& .MuiAccordionSummary-content': {
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                },
              }}
            >
              <VisibilityIcon sx={{ color: '#00D9FF', fontSize: 20 }} />
              <Typography sx={{ color: '#fff', fontWeight: 500 }}>Visual Settings</Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ pt: 0 }}>
              {/* Theme Selection */}
              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel id="theme-select-label" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Theme
                </InputLabel>
                <Select
                  labelId="theme-select-label"
                  id="theme-select"
                  value={currentTheme}
                  label="Theme"
                  onChange={handleThemeChange}
                  size="small"
                  sx={{
                    color: '#fff',
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 217, 255, 0.3)',
                    },
                    '&:hover .MuiOutlinedInput-notchedOutline': {
                      borderColor: 'rgba(0, 217, 255, 0.5)',
                    },
                    '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#00D9FF',
                    },
                    '& .MuiSvgIcon-root': {
                      color: '#00D9FF',
                    },
                  }}
                >
                  <MenuItem value="dark">Dark (Cyan)</MenuItem>
                  <MenuItem value="halloween">🎃 Halloween (Orange & Purple)</MenuItem>
                  <MenuItem value="purple">Purple</MenuItem>
                  <MenuItem value="blue">Blue</MenuItem>
                  <MenuItem value="green">Green</MenuItem>
                </Select>
              </FormControl>

              {/* 3D Background Toggle */}
              <Box sx={{ mb: 2 }}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={threeEnabled}
                      onChange={handleThreeToggle}
                      sx={{
                        color: 'rgba(0, 217, 255, 0.5)',
                        '&.Mui-checked': {
                          color: '#00D9FF',
                        },
                      }}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500 }}>
                        3D Background
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                        Enable animated 3D effects
                      </Typography>
                    </Box>
                  }
                />
              </Box>

              <Divider sx={{ borderColor: 'rgba(0, 217, 255, 0.15)', my: 2 }} />

              {/* Disable Fancy Effects */}
              <Box>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={!fancyEffects}
                      onChange={(e) => handleFancyEffectsToggle({ target: { checked: !e.target.checked } })}
                      sx={{
                        color: 'rgba(124, 77, 255, 0.5)',
                        '&.Mui-checked': {
                          color: '#7C4DFF',
                        },
                      }}
                    />
                  }
                  label={
                    <Box>
                      <Typography variant="body2" sx={{ color: '#fff', fontWeight: 500 }}>
                        Disable Fancy Effects
                      </Typography>
                      <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                        Maximize performance for low-spec hardware
                      </Typography>
                    </Box>
                  }
                />
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Performance Category */}
          <Accordion
            expanded={expandedCategory === 'performance'}
            onChange={handleCategoryChange('performance')}
            sx={{
              bgcolor: 'rgba(124, 77, 255, 0.05)',
              borderRadius: 1,
              mb: 1,
              '&:before': { display: 'none' },
              border: '1px solid rgba(124, 77, 255, 0.2)',
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon sx={{ color: '#7C4DFF' }} />}
              sx={{
                '& .MuiAccordionSummary-content': {
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                },
              }}
            >
              <TuneIcon sx={{ color: '#7C4DFF', fontSize: 20 }} />
              <Typography sx={{ color: '#fff', fontWeight: 500 }}>Performance Monitor</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 2 }}>
                Real-time performance metrics
              </Typography>
              
              {/* FPS Monitor */}
              <Box sx={{ mb: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                  <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                    Frame Rate (FPS)
                  </Typography>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: performanceMetrics.fps >= 55 ? '#00E096' : 
                             performanceMetrics.fps >= 30 ? '#FFAA00' : '#FF3D71',
                      fontWeight: 700,
                    }}
                  >
                    {performanceMetrics.fps} FPS
                  </Typography>
                </Box>
                <Box
                  sx={{
                    width: '100%',
                    height: 6,
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                    borderRadius: 1,
                    overflow: 'hidden',
                  }}
                >
                  <Box
                    sx={{
                      width: `${Math.min((performanceMetrics.fps / 60) * 100, 100)}%`,
                      height: '100%',
                      bgcolor: performanceMetrics.fps >= 55 ? '#00E096' : 
                               performanceMetrics.fps >= 30 ? '#FFAA00' : '#FF3D71',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </Box>
              </Box>

              {/* Memory Usage */}
              {performance.memory && (
                <Box sx={{ mb: 2 }}>
                  <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                    <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                      Memory Usage
                    </Typography>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: performanceMetrics.memory < 100 ? '#00E096' : 
                               performanceMetrics.memory < 200 ? '#FFAA00' : '#FF3D71',
                        fontWeight: 700,
                      }}
                    >
                      {performanceMetrics.memory} MB
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      width: '100%',
                      height: 6,
                      bgcolor: 'rgba(255, 255, 255, 0.1)',
                      borderRadius: 1,
                      overflow: 'hidden',
                    }}
                  >
                    <Box
                      sx={{
                        width: `${Math.min((performanceMetrics.memory / 250) * 100, 100)}%`,
                        height: '100%',
                        bgcolor: performanceMetrics.memory < 100 ? '#00E096' : 
                                 performanceMetrics.memory < 200 ? '#FFAA00' : '#FF3D71',
                        transition: 'width 0.3s ease',
                      }}
                    />
                  </Box>
                </Box>
              )}

              {/* Render Time */}
              <Box sx={{ mb: 1 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center">
                  <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                    Avg Frame Time
                  </Typography>
                  <Typography 
                    variant="body2" 
                    sx={{ 
                      color: performanceMetrics.renderTime < 20 ? '#00E096' : 
                             performanceMetrics.renderTime < 35 ? '#FFAA00' : '#FF3D71',
                      fontWeight: 700,
                    }}
                  >
                    {performanceMetrics.renderTime}ms
                  </Typography>
                </Box>
              </Box>

              <Divider sx={{ borderColor: 'rgba(124, 77, 255, 0.2)', my: 2 }} />

              {/* Performance Tips */}
              <Box 
                sx={{ 
                  p: 1.5, 
                  bgcolor: 'rgba(0, 217, 255, 0.05)',
                  borderRadius: 1,
                  border: '1px solid rgba(0, 217, 255, 0.2)',
                }}
              >
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.8)', fontWeight: 600, display: 'block', mb: 0.5 }}>
                  💡 Optimization Tips:
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)', display: 'block' }}>
                  • Disable 3D Background for +10-15 FPS
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)', display: 'block' }}>
                  • Disable Fancy Effects for +5-10 FPS
                </Typography>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)', display: 'block' }}>
                  • Close unused browser tabs
                </Typography>
              </Box>

              {/* Performance Status */}
              <Box sx={{ mt: 2, textAlign: 'center' }}>
                <Typography 
                  variant="caption" 
                  sx={{ 
                    color: performanceMetrics.fps >= 55 ? '#00E096' : 
                           performanceMetrics.fps >= 30 ? '#FFAA00' : '#FF3D71',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: 1,
                  }}
                >
                  {performanceMetrics.fps >= 55 ? '✓ Optimal Performance' : 
                   performanceMetrics.fps >= 30 ? '⚠ Moderate Performance' : '⚠ Low Performance'}
                </Typography>
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* System Info Category */}
          <Accordion
            expanded={expandedCategory === 'info'}
            onChange={handleCategoryChange('info')}
            sx={{
              bgcolor: 'rgba(255, 255, 255, 0.02)',
              borderRadius: 1,
              '&:before': { display: 'none' },
              border: '1px solid rgba(255, 255, 255, 0.1)',
            }}
          >
            <AccordionSummary
              expandIcon={<ExpandMoreIcon sx={{ color: 'rgba(255, 255, 255, 0.7)' }} />}
              sx={{
                '& .MuiAccordionSummary-content': {
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                },
              }}
            >
              <InfoIcon sx={{ color: 'rgba(255, 255, 255, 0.7)', fontSize: 20 }} />
              <Typography sx={{ color: '#fff', fontWeight: 500 }}>System Info</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box>
                  <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                    Version
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#fff' }}>
                    1.0.0
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                    Branch
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#fff' }}>
                    meral/taluy_gui_web
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                    ROS Bridge
                  </Typography>
                  <Typography variant="body2" sx={{ color: '#fff' }}>
                    ws://localhost:9090
                  </Typography>
                </Box>
              </Box>
            </AccordionDetails>
          </Accordion>

          {/* Info Text */}
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              mt: 3,
              color: 'rgba(255, 255, 255, 0.4)',
              textAlign: 'center',
            }}
          >
            ITU AUV Web Control Panel
          </Typography>
        </Box>
      </Drawer>
    </>
  );
};

export default VisualSettings;

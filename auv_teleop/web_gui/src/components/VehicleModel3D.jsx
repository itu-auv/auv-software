import React, { Suspense, useRef, useState } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  IconButton, 
  Tooltip,
  FormControlLabel,
  Switch,
  Slider,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ViewInArIcon from '@mui/icons-material/ViewInAr';
import * as THREE from 'three';

function STLModel({ url, color = '#00D9FF', autoRotate = false }) {
  const meshRef = useRef();
  const geometry = useLoader(STLLoader, url);

  // Center the geometry
  React.useEffect(() => {
    if (geometry) {
      geometry.center();
    }
  }, [geometry]);

  // Auto-rotate animation
  useFrame(() => {
    if (meshRef.current && autoRotate) {
      meshRef.current.rotation.z += 0.005;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} rotation={[Math.PI / -2, 0, 0]}>
      <meshStandardMaterial 
        color={color} 
        metalness={0.6} 
        roughness={0.3}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

function VehicleModel3D({ modelPath = '/body.stl' }) {
  const [autoRotate, setAutoRotate] = useState(true);
  const [modelColor, setModelColor] = useState('#00D9FF');
  const [gridVisible, setGridVisible] = useState(true);
  const [key, setKey] = useState(0);

  const handleReset = () => {
    setKey(prev => prev + 1);
    setAutoRotate(true);
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <ViewInArIcon color="primary" />
            <Typography variant="h6">Vehicle 3D Model</Typography>
          </Box>
          <Tooltip title="Reset View">
            <IconButton onClick={handleReset} size="small" color="primary">
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>

        {/* 3D Canvas */}
        <Box 
          sx={{ 
            width: '100%', 
            height: 400, 
            bgcolor: 'rgba(0, 0, 0, 0.3)', 
            borderRadius: 2,
            overflow: 'hidden',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Canvas key={key}>
            <PerspectiveCamera makeDefault position={[2, 2, 2]} />
            <OrbitControls 
              enableDamping 
              dampingFactor={0.05}
              autoRotate={autoRotate}
              autoRotateSpeed={2}
            />
            
            {/* Lighting */}
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <directionalLight position={[-10, -10, -5]} intensity={0.5} />
            <pointLight position={[0, 5, 0]} intensity={0.5} />
            
            {/* Grid */}
            {gridVisible && (
              <Grid 
                args={[10, 10]} 
                cellSize={0.5} 
                cellThickness={0.5} 
                cellColor="#6f6f6f"
                sectionSize={1}
                sectionThickness={1}
                sectionColor="#00D9FF"
                fadeDistance={10}
                fadeStrength={1}
                followCamera={false}
                infiniteGrid
              />
            )}
            
            {/* STL Model */}
            <Suspense fallback={null}>
              <STLModel url={modelPath} color={modelColor} autoRotate={false} />
            </Suspense>
          </Canvas>
        </Box>

        {/* Controls */}
        <Box mt={2} display="flex" flexDirection="column" gap={1.5}>
          <FormControlLabel
            control={
              <Switch 
                checked={autoRotate} 
                onChange={(e) => setAutoRotate(e.target.checked)}
                color="primary"
              />
            }
            label="Auto Rotate"
          />
          
          <FormControlLabel
            control={
              <Switch 
                checked={gridVisible} 
                onChange={(e) => setGridVisible(e.target.checked)}
                color="primary"
              />
            }
            label="Show Grid"
          />

          <Box>
            <Typography variant="caption" color="text.secondary" gutterBottom>
              Model Color
            </Typography>
            <Box display="flex" gap={1} mt={0.5}>
              {['#00D9FF', '#7C4DFF', '#F25912', '#00E096', '#FFFFFF'].map((color) => (
                <Box
                  key={color}
                  onClick={() => setModelColor(color)}
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: color,
                    borderRadius: 1,
                    cursor: 'pointer',
                    border: modelColor === color ? '3px solid #fff' : '1px solid rgba(255,255,255,0.2)',
                    transition: 'all 0.2s',
                    '&:hover': {
                      transform: 'scale(1.1)',
                    },
                  }}
                />
              ))}
            </Box>
          </Box>
        </Box>

        <Typography variant="caption" color="text.secondary" mt={2} display="block">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </Typography>
      </CardContent>
    </Card>
  );
}

export default VehicleModel3D;

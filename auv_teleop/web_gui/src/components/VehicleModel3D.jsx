import React, { Suspense, useRef, useState } from 'react';
import { Canvas, useLoader, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { RotateCcw, Box as BoxIcon } from 'lucide-react';
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
    <Card className="glass-card">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-2">
            <BoxIcon className="w-5 h-5 text-primary" />
            <CardTitle>Vehicle 3D Model</CardTitle>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="icon" onClick={handleReset} className="h-8 w-8">
                  <RotateCcw className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Reset View</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </CardHeader>
      <CardContent>
        {/* 3D Canvas */}
        <div className="w-full h-[400px] bg-black/30 rounded-lg overflow-hidden border border-white/10">
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
        </div>

        {/* Controls */}
        <div className="mt-4 flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <label htmlFor="auto-rotate" className="text-sm">Auto Rotate</label>
            <Switch
              id="auto-rotate"
              checked={autoRotate}
              onCheckedChange={setAutoRotate}
            />
          </div>

          <div className="flex items-center justify-between">
            <label htmlFor="grid-visible" className="text-sm">Show Grid</label>
            <Switch
              id="grid-visible"
              checked={gridVisible}
              onCheckedChange={setGridVisible}
            />
          </div>

          <div>
            <p className="text-xs text-muted-foreground mb-2">Model Color</p>
            <div className="flex gap-2">
              {['#00D9FF', '#7C4DFF', '#F25912', '#00E096', '#FFFFFF'].map((color) => (
                <button
                  key={color}
                  onClick={() => setModelColor(color)}
                  className="w-8 h-8 rounded cursor-pointer transition-transform hover:scale-110"
                  style={{
                    backgroundColor: color,
                    border: modelColor === color ? '3px solid #fff' : '1px solid rgba(255,255,255,0.2)',
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        <p className="text-xs text-muted-foreground mt-4">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </p>
      </CardContent>
    </Card>
  );
}

export default VehicleModel3D;

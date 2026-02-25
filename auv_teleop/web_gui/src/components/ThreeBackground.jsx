import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial } from '@react-three/drei';

// Animated central sphere with distortion
function CentralSphere() {
  const meshRef = useRef();

  useFrame((state) => {
    if (meshRef.current) {
      const time = state.clock.getElapsedTime();
      meshRef.current.rotation.x = time * 0.15;
      meshRef.current.rotation.y = time * 0.2;
      meshRef.current.position.y = Math.sin(time * 0.5) * 0.3;
    }
  });

  return (
    <Sphere ref={meshRef} args={[1.2, 128, 128]} scale={1.8}>
      <MeshDistortMaterial
        color="#00D9FF"
        attach="material"
        distort={0.4}
        speed={2}
        roughness={0.1}
        metalness={0.9}
        emissive="#00D9FF"
        emissiveIntensity={0.1}
      />
    </Sphere>
  );
}

// Rotating geometric rings
function GeometricRings() {
  const group = useRef();

  useFrame((state) => {
    if (group.current) {
      const time = state.clock.getElapsedTime();
      group.current.rotation.x = time * 0.1;
      group.current.rotation.y = time * 0.15;
      group.current.rotation.z = time * 0.05;
    }
  });

  return (
    <group ref={group}>
      <mesh rotation={[0, 0, 0]}>
        <torusGeometry args={[3, 0.08, 16, 100]} />
        <meshStandardMaterial color="#7C4DFF" emissive="#7C4DFF" emissiveIntensity={0.2} metalness={0.8} />
      </mesh>
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[3.2, 0.08, 16, 100]} />
        <meshStandardMaterial color="#00D9FF" emissive="#00D9FF" emissiveIntensity={0.2} metalness={0.8} />
      </mesh>
      <mesh rotation={[0, Math.PI / 2, 0]}>
        <torusGeometry args={[3.4, 0.08, 16, 100]} />
        <meshStandardMaterial color="#FF00FF" emissive="#FF00FF" emissiveIntensity={0.1} metalness={0.8} />
      </mesh>
    </group>
  );
}

// Floating particle field
function ParticleField() {
  const count = 200;
  const mesh = useRef();

  const particles = useMemo(() => {
    const temp = [];
    for (let i = 0; i < count; i++) {
      const x = (Math.random() - 0.5) * 25;
      const y = (Math.random() - 0.5) * 25;
      const z = (Math.random() - 0.5) * 25;
      const scale = Math.random() * 0.5 + 0.1;
      temp.push({ position: [x, y, z], scale, key: i });
    }
    return temp;
  }, [count]);

  useFrame((state) => {
    if (mesh.current) {
      mesh.current.rotation.y = state.clock.getElapsedTime() * 0.03;
    }
  });

  return (
    <group ref={mesh}>
      {particles.map((particle) => (
        <mesh key={particle.key} position={particle.position} scale={particle.scale}>
          <sphereGeometry args={[0.15, 8, 8]} />
          <meshStandardMaterial
            color="#7C4DFF"
            emissive="#7C4DFF"
            emissiveIntensity={0.3}
            transparent
            opacity={0.3}
          />
        </mesh>
      ))}
    </group>
  );
}

// Orbiting polyhedrons
function OrbitingShapes() {
  const group = useRef();

  useFrame((state) => {
    if (group.current) {
      group.current.rotation.y = state.clock.getElapsedTime() * 0.3;
    }
  });

  return (
    <group ref={group}>
      <mesh position={[4, 0, 0]}>
        <icosahedronGeometry args={[0.5, 0]} />
        <meshStandardMaterial color="#00D9FF" emissive="#00D9FF" emissiveIntensity={0.2} wireframe />
      </mesh>
      <mesh position={[-4, 0, 0]}>
        <octahedronGeometry args={[0.5, 0]} />
        <meshStandardMaterial color="#7C4DFF" emissive="#7C4DFF" emissiveIntensity={0.2} wireframe />
      </mesh>
      <mesh position={[0, 4, 0]}>
        <tetrahedronGeometry args={[0.5, 0]} />
        <meshStandardMaterial color="#FF00FF" emissive="#FF00FF" emissiveIntensity={0.2} wireframe />
      </mesh>
      <mesh position={[0, -4, 0]}>
        <dodecahedronGeometry args={[0.5, 0]} />
        <meshStandardMaterial color="#00FF88" emissive="#00FF88" emissiveIntensity={0.2} wireframe />
      </mesh>
    </group>
  );
}

function ThreeBackground({ enabled = true }) {
  if (!enabled) return null;

  return (
    <div
      className="fixed top-0 left-0 w-screen h-screen -z-10 pointer-events-none opacity-25"
    >
      <Canvas
        camera={{ position: [0, 0, 10], fov: 75 }}
        style={{ width: '100%', height: '100%' }}
      >
        {/* Background gradient */}
        <color attach="background" args={['#000510']} />
        <fog attach="fog" args={['#000510', 8, 25]} />

        {/* Lighting - reduced intensity */}
        <ambientLight intensity={0.2} />
        <pointLight position={[10, 10, 10]} intensity={0.8} color="#00D9FF" />
        <pointLight position={[-10, -10, -10]} intensity={0.8} color="#7C4DFF" />
        <spotLight
          position={[0, 15, 0]}
          angle={0.5}
          penumbra={1}
          intensity={0.5}
          color="#00D9FF"
          castShadow
        />

        {/* 3D Elements */}
        <CentralSphere />
        <GeometricRings />
        <ParticleField />
        <OrbitingShapes />
      </Canvas>
    </div>
  );
}

export default ThreeBackground;

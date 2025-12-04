import React, { Suspense, useState } from 'react';
import { Waves, Monitor, Loader2, ChevronRight } from 'lucide-react';
import VehicleModel3DBackground from './VehicleModel3DBackground';

function StartScreen({ onSelectMode }) {
  const [modelError, setModelError] = useState(false);

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-black relative overflow-hidden p-8">
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black/50 to-black pointer-events-none" />

      {/* Subtle grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: '60px 60px'
        }}
      />

      {/* 3D Model Background */}
      {!modelError && (
        <Suspense
          fallback={
            <div className="absolute top-4 left-4 flex items-center gap-2 text-white/30">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span className="text-xs">Loading 3D Model...</span>
            </div>
          }
        >
          <VehicleModel3DBackground color="#ffffff" />
        </Suspense>
      )}

      <div className="max-w-3xl w-full relative z-10">
        {/* Title */}
        <div className="text-center mb-16">
          <p className="text-xs uppercase tracking-[0.3em] text-white/30 mb-4 font-medium">
            Istanbul Technical University
          </p>
          <h1 className="text-5xl md:text-6xl font-light mb-4 tracking-tight text-white">
            AUV Control
          </h1>
          <div className="w-16 h-px bg-white/20 mx-auto mb-4" />
          <p className="text-sm text-white/40 font-light">
            Select operation mode to continue
          </p>
        </div>

        {/* Mode Selection Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Pool Test Mode */}
          <button
            className="group text-left p-8 rounded-2xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-xl transition-all duration-300 hover:bg-white/[0.05] hover:border-white/[0.15] hover:scale-[1.02]"
            onClick={() => onSelectMode('pool')}
          >
            <div className="w-12 h-12 mb-6 flex items-center justify-center rounded-xl bg-white/5 border border-white/10 group-hover:bg-white/10 transition-colors">
              <Waves className="w-5 h-5 text-white/60 group-hover:text-white/80 transition-colors" />
            </div>

            <h2 className="text-lg font-medium mb-2 text-white/90 group-hover:text-white transition-colors">
              Pool Test
            </h2>

            <p className="text-sm text-white/40 mb-4 leading-relaxed">
              Real hardware testing in pool environment
            </p>

            <div className="flex items-center justify-between">
              <span className="text-xs text-white/20 italic">
                Coming soon
              </span>
              <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white/50 group-hover:translate-x-1 transition-all" />
            </div>
          </button>

          {/* Simulation Mode */}
          <button
            className="group text-left p-8 rounded-2xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-xl transition-all duration-300 hover:bg-white/[0.05] hover:border-white/[0.15] hover:scale-[1.02]"
            onClick={() => onSelectMode('simulation')}
          >
            <div className="w-12 h-12 mb-6 flex items-center justify-center rounded-xl bg-white/5 border border-white/10 group-hover:bg-white/10 transition-colors">
              <Monitor className="w-5 h-5 text-white/60 group-hover:text-white/80 transition-colors" />
            </div>

            <h2 className="text-lg font-medium mb-2 text-white/90 group-hover:text-white transition-colors">
              Simulation
            </h2>

            <p className="text-sm text-white/40 leading-relaxed">
              Virtual testing with Gazebo simulation
            </p>

            <div className="flex items-center justify-end mt-4">
              <ChevronRight className="w-4 h-4 text-white/20 group-hover:text-white/50 group-hover:translate-x-1 transition-all" />
            </div>
          </button>
        </div>

        {/* Footer */}
        <div className="mt-16 text-center">
          <p className="text-[10px] uppercase tracking-[0.2em] text-white/20">
            Autonomous Underwater Vehicle Team
          </p>
        </div>
      </div>
    </div>
  );
}

export default StartScreen;

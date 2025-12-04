import React from 'react';
import { Battery, BatteryWarning, BatteryFull, BatteryLow, Zap } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

function BatteryStatus({
  voltage,
  current,
  power,
  powerHistory,
  powerTopicName,
  lastPowerUpdate,
  powerSource
}) {
  const getStatusColor = () => {
    if (voltage === 0) return 'text-white/30';
    if (voltage < 14.8) return 'text-red-400';
    if (voltage < 15.2) return 'text-amber-400';
    return 'text-emerald-400';
  };

  const getStatusLabel = () => {
    if (voltage === 0) return 'NO DATA';
    if (voltage < 14.8) return 'CRITICAL';
    if (voltage < 15.2) return 'LOW';
    return 'GOOD';
  };

  const getBatteryIcon = () => {
    if (voltage === 0) return Battery;
    if (voltage < 14.8) return BatteryWarning;
    if (voltage < 15.2) return BatteryLow;
    return BatteryFull;
  };

  const BatteryIcon = getBatteryIcon();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span className="flex items-center gap-2">
            <BatteryIcon className="w-4 h-4 text-white/50" />
            Battery
          </span>
          <span className={`text-xs font-medium ${getStatusColor()}`}>
            {getStatusLabel()}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {voltage > 0 ? (
          <>
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wider text-white/30">Voltage</p>
                <p className="text-xl font-light tabular-nums text-white">
                  {voltage.toFixed(1)}
                  <span className="text-xs text-white/30 ml-0.5">V</span>
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wider text-white/30">Current</p>
                <p className="text-xl font-light tabular-nums text-white">
                  {current.toFixed(1)}
                  <span className="text-xs text-white/30 ml-0.5">A</span>
                </p>
              </div>
              <div className="space-y-1">
                <p className="text-[10px] uppercase tracking-wider text-white/30">Power</p>
                <p className="text-xl font-light tabular-nums text-white">
                  {power.toFixed(0)}
                  <span className="text-xs text-white/30 ml-0.5">W</span>
                </p>
              </div>
            </div>

            {/* Power History Graph */}
            {powerHistory.length > 1 && (
              <div className="pt-2">
                <p className="text-[10px] uppercase tracking-wider text-white/30 mb-2">Power History</p>
                <div className="relative h-12 bg-white/[0.03] rounded-lg p-2 border border-white/[0.05]">
                  <svg width="100%" height="100%" className="block">
                    <polyline
                      fill="none"
                      stroke="rgba(255,255,255,0.4)"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      points={powerHistory.map((point, index) => {
                        const x = (index / (powerHistory.length - 1)) * 100;
                        const maxPower = Math.max(...powerHistory.map(p => p.power), 1);
                        const y = 100 - (point.power / maxPower) * 80;
                        return `${x}%,${y}%`;
                      }).join(' ')}
                    />
                  </svg>
                </div>
              </div>
            )}

            <div className="flex items-center gap-2 text-[10px] text-white/20">
              <Zap className="w-3 h-3" />
              {powerSource === 'simulation' ? 'Simulation' : 'Hardware'}
            </div>
          </>
        ) : (
          <div className="text-center py-4">
            <Battery className="w-8 h-8 mx-auto mb-2 text-white/20" />
            <p className="text-sm text-white/40">Waiting for data...</p>
            <p className="text-[10px] mt-1 text-white/20">{powerTopicName || 'Searching...'}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default BatteryStatus;

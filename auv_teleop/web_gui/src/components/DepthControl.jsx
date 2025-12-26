import React from 'react';
import { Waves, ArrowDown } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

function DepthControl({ depth, setDepth, setDepthService, connected }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Waves className="w-4 h-4 text-white/50" />
          Depth
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <p className="text-[10px] uppercase tracking-wider text-white/30">Target Depth</p>
          <div className="flex items-center gap-3">
            <Input
              type="number"
              value={depth}
              onChange={(e) => setDepth(parseFloat(e.target.value) || 0)}
              min={-3.0}
              max={0.0}
              step={0.1}
              className="text-2xl font-light h-14 text-center bg-white/[0.02]"
            />
            <span className="text-white/30 text-sm">m</span>
          </div>
          <div className="flex justify-between text-[10px] text-white/20 px-1">
            <span>Surface: 0m</span>
            <span>Max: -3m</span>
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2">
          {[0, -1, -2, -3].map((d) => (
            <Button
              key={d}
              variant={depth === d ? "default" : "secondary"}
              size="sm"
              onClick={() => setDepth(d)}
              className="text-xs"
            >
              {d}m
            </Button>
          ))}
        </div>

        <Button
          className="w-full"
          onClick={setDepthService}
          disabled={!connected}
        >
          <ArrowDown className="w-4 h-4 mr-2" />
          Set Depth
        </Button>
      </CardContent>
    </Card>
  );
}

export default DepthControl;

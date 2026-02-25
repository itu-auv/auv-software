import React from 'react';
import { Crosshair, Rocket, Circle } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

function MissionControls({
  connected,
  launchTorpedo1,
  launchTorpedo2,
  dropBall
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Crosshair className="w-4 h-4 text-white/50" />
          Mission
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="destructive"
            size="sm"
            onClick={launchTorpedo1}
            disabled={!connected}
          >
            <Rocket className="w-3 h-3 mr-1.5" />
            Torpedo 1
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={launchTorpedo2}
            disabled={!connected}
          >
            <Rocket className="w-3 h-3 mr-1.5" />
            Torpedo 2
          </Button>
        </div>
        <Button
          variant="warning"
          className="w-full"
          onClick={dropBall}
          disabled={!connected}
        >
          <Circle className="w-4 h-4 mr-2" />
          Drop Ball
        </Button>
      </CardContent>
    </Card>
  );
}

export default MissionControls;

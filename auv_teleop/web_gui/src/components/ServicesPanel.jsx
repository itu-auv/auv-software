import React, { useState } from 'react';
import { Settings, MapPin, Trash2, RefreshCw, Radio } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';

function ServicesPanel({
  connected,
  startLocalization,
  enableDVL,
  disableDVL,
  clearObjects,
  resetPose
}) {
  const [dvlEnabled, setDvlEnabled] = useState(false);

  const handleDvlToggle = (checked) => {
    setDvlEnabled(checked);
    if (checked) {
      enableDVL();
    } else {
      disableDVL();
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Settings className="w-4 h-4 text-white/50" />
          Services
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Button
          className="w-full"
          onClick={startLocalization}
          disabled={!connected}
        >
          <MapPin className="w-4 h-4 mr-2" />
          Start Localization
        </Button>

        <div className="flex items-center justify-between p-3 rounded-xl border border-white/[0.06] bg-white/[0.02]">
          <div className="flex items-center gap-2">
            <Radio className={`w-4 h-4 ${dvlEnabled ? 'text-emerald-400' : 'text-white/30'}`} />
            <span className="text-sm text-white/70">DVL Sensor</span>
          </div>
          <Switch
            checked={dvlEnabled}
            onCheckedChange={handleDvlToggle}
            disabled={!connected}
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={clearObjects}
            disabled={!connected}
          >
            <Trash2 className="w-3 h-3 mr-1.5" />
            Clear
          </Button>
          <Button
            variant="secondary"
            size="sm"
            onClick={resetPose}
            disabled={!connected}
          >
            <RefreshCw className="w-3 h-3 mr-1.5" />
            Reset
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default ServicesPanel;

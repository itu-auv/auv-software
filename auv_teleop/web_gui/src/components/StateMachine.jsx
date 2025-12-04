import React from 'react';
import { Play, StopCircle, Bot } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';

function StateMachine({
  connected,
  testMode,
  setTestMode,
  smachRunning,
  setSmachRunning,
  selectedStates,
  setSelectedStates
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-white/50" />
          State Machine (SMACH)
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-3 p-3 rounded-xl border border-white/[0.06] bg-white/[0.02]">
            <span className="text-sm text-white/70">Test Mode</span>
            <Switch
              checked={testMode}
              onCheckedChange={setTestMode}
              disabled={!connected || smachRunning}
            />
          </div>

          <Button
            variant="success"
            onClick={() => setSmachRunning(true)}
            disabled={!connected || smachRunning}
          >
            <Play className="w-4 h-4 mr-2" />
            Launch
          </Button>

          <Button
            variant="destructive"
            onClick={() => setSmachRunning(false)}
            disabled={!connected || !smachRunning}
          >
            <StopCircle className="w-4 h-4 mr-2" />
            Stop
          </Button>
        </div>

        {testMode && (
          <div className="p-4 rounded-xl border border-white/[0.06] bg-white/[0.02]">
            <p className="text-[10px] uppercase tracking-wider text-white/30 mb-3">Select Test States</p>
            <div className="flex flex-wrap gap-3">
              {Object.keys(selectedStates).map((state) => (
                <div key={state} className="flex items-center gap-2 p-2 rounded-lg bg-white/[0.03] border border-white/[0.05]">
                  <Switch
                    checked={selectedStates[state]}
                    onCheckedChange={(checked) => setSelectedStates(prev => ({
                      ...prev,
                      [state]: checked
                    }))}
                    disabled={!connected || smachRunning}
                  />
                  <label className="text-sm text-white/60 capitalize">{state}</label>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default StateMachine;

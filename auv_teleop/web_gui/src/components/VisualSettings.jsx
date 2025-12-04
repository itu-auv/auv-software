import React, { useState, useEffect } from 'react';
import { Settings, X, Gauge } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select';

const VisualSettings = ({
  currentTheme,
  setTheme,
  fancyEffects,
  setFancyEffects
}) => {
  const [open, setOpen] = useState(false);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId;

    const measurePerformance = () => {
      frameCount++;
      const currentTime = performance.now();
      const elapsed = currentTime - lastTime;

      if (elapsed >= 1000) {
        setFps(Math.round((frameCount * 1000) / elapsed));
        frameCount = 0;
        lastTime = currentTime;
      }

      animationId = requestAnimationFrame(measurePerformance);
    };

    animationId = requestAnimationFrame(measurePerformance);
    return () => cancelAnimationFrame(animationId);
  }, []);

  // Apply theme class to root element
  useEffect(() => {
    const root = document.documentElement;
    root.classList.remove('theme-halloween');
    if (currentTheme === 'halloween') {
      root.classList.add('theme-halloween');
    }
  }, [currentTheme]);

  return (
    <>
      {/* Floating Button */}
      <Button
        size="icon"
        variant="outline"
        onClick={() => setOpen(true)}
        className="fixed right-4 top-1/2 -translate-y-1/2 z-50"
      >
        <Settings className="w-4 h-4" />
      </Button>

      {/* Drawer */}
      {open && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50"
            onClick={() => setOpen(false)}
          />

          {/* Panel */}
          <div className="fixed right-0 top-0 bottom-0 w-80 bg-background border-l z-50 overflow-y-auto">
            <div className="p-6 space-y-6">
              {/* Header */}
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Settings</h2>
                <Button size="icon" variant="ghost" onClick={() => setOpen(false)}>
                  <X className="w-4 h-4" />
                </Button>
              </div>

              <div className="space-y-4">
                {/* Theme Select */}
                <div className="space-y-2">
                  <label className="text-sm font-medium">Theme</label>
                  <Select value={currentTheme} onValueChange={setTheme}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="halloween">Halloween</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {/* Fancy Effects */}
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <label className="text-sm font-medium">Animations</label>
                    <p className="text-xs text-muted-foreground">Enable fancy effects</p>
                  </div>
                  <Switch
                    checked={fancyEffects}
                    onCheckedChange={setFancyEffects}
                  />
                </div>
              </div>

              {/* Performance */}
              <div className="pt-4 border-t">
                <div className="flex items-center gap-2 mb-3">
                  <Gauge className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm font-medium">Performance</span>
                </div>
                <div className="flex justify-between items-center p-3 rounded-lg bg-secondary">
                  <span className="text-sm text-muted-foreground">Frame Rate</span>
                  <span className={`text-sm font-mono font-bold ${
                    fps >= 55 ? 'text-green-500' : fps >= 30 ? 'text-yellow-500' : 'text-red-500'
                  }`}>
                    {fps} FPS
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default VisualSettings;

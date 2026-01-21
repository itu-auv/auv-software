import React from 'react';
import { RefreshCw, Loader2, Wifi, WifiOff } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

function Header({ connected, connecting, connectToROS }) {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.06] bg-black/80 backdrop-blur-xl">
      <div className="container flex h-16 items-center px-6">
        <div className="flex items-center gap-4 flex-1">
          <img
            src="/logo.png"
            alt="ITU AUV Logo"
            className="h-8 w-auto opacity-90"
            onError={(e) => e.target.style.display = 'none'}
          />
          <div className="flex flex-col">
            <span className="font-medium text-base tracking-tight text-white/90">
              ITU AUV
            </span>
            <span className="text-[10px] text-white/30 font-medium uppercase tracking-[0.15em]">
              Control Panel
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {connecting && (
            <Loader2 className="w-4 h-4 animate-spin text-white/30" />
          )}

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium ${
            connected
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
              : 'bg-white/5 text-white/40 border border-white/10'
          }`}>
            {connected ? (
              <Wifi className="w-3 h-3" />
            ) : (
              <WifiOff className="w-3 h-3" />
            )}
            {connected ? 'Connected' : 'Disconnected'}
          </div>

          {!connected && (
            <Button
              variant="outline"
              size="sm"
              onClick={connectToROS}
              disabled={connecting}
              className="text-xs"
            >
              <RefreshCw className={`w-3 h-3 mr-2 ${connecting ? 'animate-spin' : ''}`} />
              Connect
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;

import * as React from "react";
import * as SwitchPrimitives from "@radix-ui/react-switch";
import { cn } from "@/lib/utils";

const Switch = React.forwardRef(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className={cn(
      "peer inline-flex h-5 w-9 shrink-0 cursor-pointer items-center rounded-full border border-white/10 transition-all duration-200 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-40 data-[state=checked]:bg-white data-[state=unchecked]:bg-white/10",
      className
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        "pointer-events-none block h-4 w-4 rounded-full shadow-sm transition-all duration-200 data-[state=checked]:translate-x-4 data-[state=unchecked]:translate-x-0.5 data-[state=checked]:bg-black data-[state=unchecked]:bg-white/60"
      )}
    />
  </SwitchPrimitives.Root>
));
Switch.displayName = SwitchPrimitives.Root.displayName;

export { Switch };

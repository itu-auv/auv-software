import * as React from "react";
import { cn } from "@/lib/utils";

const Input = React.forwardRef(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      className={cn(
        "flex h-10 w-full rounded-xl border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-white placeholder:text-white/30 focus-visible:outline-none focus-visible:border-white/30 focus-visible:bg-white/[0.05] disabled:cursor-not-allowed disabled:opacity-40 transition-all duration-200",
        className
      )}
      ref={ref}
      {...props}
    />
  );
});
Input.displayName = "Input";

export { Input };

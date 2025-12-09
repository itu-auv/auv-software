import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-xl text-sm font-medium ring-offset-background transition-all duration-200 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-white/30 disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        default: "bg-white text-black hover:bg-white/90 shadow-lg shadow-white/10",
        destructive: "bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30",
        outline: "border border-white/20 bg-transparent hover:bg-white/5 text-white/80 hover:text-white",
        secondary: "bg-white/10 text-white/80 hover:bg-white/15 border border-white/5",
        ghost: "hover:bg-white/5 text-white/60 hover:text-white/90",
        link: "text-white/80 underline-offset-4 hover:underline hover:text-white",
        success: "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30",
        warning: "bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30",
      },
      size: {
        default: "h-10 px-5 py-2",
        sm: "h-8 rounded-lg px-3 text-xs",
        lg: "h-12 rounded-xl px-8",
        icon: "h-9 w-9 rounded-lg",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

const Button = React.forwardRef(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };

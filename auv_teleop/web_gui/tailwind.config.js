/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // Custom colors for AUV
        cyan: {
          DEFAULT: "#00D9FF",
          50: "#E5FAFF",
          100: "#CCF5FF",
          200: "#99EBFF",
          300: "#66E1FF",
          400: "#33D7FF",
          500: "#00D9FF",
          600: "#00A7CC",
          700: "#007A99",
          800: "#004D66",
          900: "#002033",
        },
        purple: {
          DEFAULT: "#7C4DFF",
          500: "#7C4DFF",
          600: "#651FFF",
          700: "#4A148C",
        },
        halloween: {
          orange: "#F25912",
          purple: "#5C3E94",
          darkPurple: "#412B6B",
          darkest: "#211832",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      animation: {
        "pulse-glow": "pulse-glow 2s ease-in-out infinite",
        "gradient-shift": "gradient-shift 3s ease infinite",
      },
      keyframes: {
        "pulse-glow": {
          "0%, 100%": {
            textShadow: "0 0 10px currentColor, 0 0 20px currentColor"
          },
          "50%": {
            textShadow: "0 0 20px currentColor, 0 0 30px currentColor, 0 0 40px currentColor"
          },
        },
        "gradient-shift": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}

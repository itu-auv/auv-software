import { createTheme } from '@mui/material/styles';

export const createHalloweenTheme = (fancyEffects = true) => createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#F25912', // Halloween Orange
      light: '#FF7A3D',
      dark: '#C44509',
    },
    secondary: {
      main: '#5C3E94', // Halloween Purple
      light: '#8067B7',
      dark: '#412B6B',
    },
    error: {
      main: '#FF3D71',
    },
    warning: {
      main: '#F25912',
    },
    success: {
      main: '#5C3E94',
    },
    info: {
      main: '#F25912',
    },
    background: {
      default: '#211832', // Dark Halloween Purple
      paper: '#412B6B', // Medium Halloween Purple
    },
    text: {
      primary: '#F25912', // Halloween Orange
      secondary: '#FF7A3D', // Light Halloween Orange
    },
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 700,
      letterSpacing: '-0.5px',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.25px',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: 'transparent',
        },
        '#root': {
          backgroundColor: 'transparent',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: fancyEffects ? {
          backgroundImage: 'linear-gradient(135deg, rgba(242, 89, 18, 0.05) 0%, rgba(92, 62, 148, 0.05) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(242, 89, 18, 0.2)',
          boxShadow: '0 8px 32px 0 rgba(242, 89, 18, 0.2)',
        } : {
          backgroundColor: 'rgba(92, 62, 148, 0.1)',
          border: '1px solid rgba(242, 89, 18, 0.2)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
        contained: fancyEffects ? {
          boxShadow: '0 4px 14px 0 rgba(242, 89, 18, 0.39)',
          '&:hover': {
            boxShadow: '0 6px 20px rgba(242, 89, 18, 0.5)',
          },
        } : {},
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: fancyEffects ? {
          backgroundImage: 'linear-gradient(135deg, rgba(242, 89, 18, 0.15) 0%, rgba(92, 62, 148, 0.15) 100%)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(242, 89, 18, 0.2)',
        } : {
          backgroundColor: 'rgba(65, 43, 107, 0.95)',
          borderBottom: '1px solid rgba(242, 89, 18, 0.2)',
        },
      },
    },
  },
});

export const createDarkTheme = (fancyEffects = true) => createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00D9FF', // Cyan - more modern
      light: '#5DFDFF',
      dark: '#00A7CC',
    },
    secondary: {
      main: '#7C4DFF', // Purple
      light: '#B47CFF',
      dark: '#3F1DCB',
    },
    error: {
      main: '#FF3D71',
    },
    warning: {
      main: '#FFAA00',
    },
    success: {
      main: '#00E096',
    },
    info: {
      main: '#00D9FF',
    },
    background: {
      default: '#0A0E27', // Deep blue-black
      paper: '#151932', // Slightly lighter blue-black
    },
    text: {
      primary: '#FFFFFF',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  shape: {
    borderRadius: 12, // More rounded corners
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h5: {
      fontWeight: 700,
      letterSpacing: '-0.5px',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '-0.25px',
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: 'transparent',
        },
        '#root': {
          backgroundColor: 'transparent',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: fancyEffects ? {
          backgroundImage: 'linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%)',
          backdropFilter: 'blur(10px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        } : {
          backgroundColor: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 8,
        },
        contained: fancyEffects ? {
          boxShadow: '0 4px 14px 0 rgba(0, 217, 255, 0.39)',
          '&:hover': {
            boxShadow: '0 6px 20px rgba(0, 217, 255, 0.5)',
          },
        } : {},
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 600,
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: fancyEffects ? {
          backgroundImage: 'linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(124, 77, 255, 0.1) 100%)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        } : {
          backgroundColor: 'rgba(21, 25, 50, 0.95)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        },
      },
    },
  },
});

// Theme factory function
export const createThemeByName = (themeName, fancyEffects = true) => {
  switch (themeName) {
    case 'halloween':
      return createHalloweenTheme(fancyEffects);
    case 'dark':
    default:
      return createDarkTheme(fancyEffects);
  }
};

// Default export for backwards compatibility
export const darkTheme = createDarkTheme(true);

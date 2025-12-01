import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import AutoRLLandingPage from '../pages/AutoRLLandingPage';

// Simple theme for landing page (lighter than main app theme)
const landingTheme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1e88e5',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    background: {
      default: '#ffffff',
      paper: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
  },
});

export default function LandingPageWrapper() {
  return (
    <ThemeProvider theme={landingTheme}>
      <CssBaseline />
      <AutoRLLandingPage />
    </ThemeProvider>
  );
}

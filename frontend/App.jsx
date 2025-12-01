import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Import page components
import Navigation from './components/Navigation.js';
import LandingPageWrapper from './components/LandingPageWrapper';
import Dashboard from './pages/Dashboard';
import ErrorBoundary from './components/ErrorBoundary';
import DeviceManager from './pages/DeviceManager';
import TaskExecution from './pages/TaskExecution';
import AITraining from './pages/AITraining';
import Analytics from './pages/Analytics';
import Marketplace from './pages/Marketplace';
import Documentation from './pages/Documentation';
import Settings from './pages/Settings';
import Profile from './pages/Profile';
import BlockchainEnhancedDashboard from './pages/BlockchainEnhancedDashboard';
import OMHIntegrationPage from './pages/OMHIntegrationPage';

// Import OMH Auth Context
import { OMHAuthProvider } from './contexts/OMHAuthContext';
import { Web3Provider } from './contexts/Web3Context';

// AutoRL Blue Theme Configuration
const autorlTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1e88e5', // Electric blue
      light: '#42a5f5',
      dark: '#1565c0',
      contrastText: '#ffffff',
    },
    secondary: {
      main: '#0288d1', // Deep blue
      light: '#03a9f4',
      dark: '#01579b',
      contrastText: '#ffffff',
    },
    info: {
      main: '#00b0ff', // Bright blue
      light: '#40c4ff',
      dark: '#0091ea',
    },
    background: {
      default: '#0a0e1a', // Deep blue-black
      paper: '#12182b', // Dark blue-gray
    },
    text: {
      primary: '#e8f4fd',
      secondary: '#90caf9',
    },
    action: {
      active: '#42a5f5',
      hover: 'rgba(30, 136, 229, 0.08)',
      selected: 'rgba(30, 136, 229, 0.16)',
      disabled: 'rgba(255, 255, 255, 0.3)',
      disabledBackground: 'rgba(255, 255, 255, 0.12)',
    },
    divider: 'rgba(30, 136, 229, 0.12)',
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 600,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 600,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 600,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(135deg, rgba(30, 136, 229, 0.05) 0%, rgba(21, 101, 192, 0.05) 100%)',
          backdropFilter: 'blur(10px)',
          borderRadius: 12,
          border: '1px solid rgba(30, 136, 229, 0.1)',
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
        contained: {
          boxShadow: '0 4px 12px rgba(30, 136, 229, 0.3)',
          '&:hover': {
            boxShadow: '0 6px 16px rgba(30, 136, 229, 0.4)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 6,
        },
      },
    },
  },
  shape: {
    borderRadius: 8,
  },
});

function App() {
  return (
    <ThemeProvider theme={autorlTheme}>
      <CssBaseline />
      <ErrorBoundary>
        <Web3Provider>
          <OMHAuthProvider>
            <div className="App">
              <Routes>
                <Route path="/" element={<LandingPageWrapper />} />
                <Route path="/app/*" element={
                  <>
                    <Navigation />
                    <Routes>
                      <Route path="dashboard" element={<Dashboard />} />
                      <Route path="devices" element={<DeviceManager />} />
                      <Route path="tasks" element={<TaskExecution />} />
                      <Route path="ai-training" element={<AITraining />} />
                      <Route path="analytics" element={<Analytics />} />
                      <Route path="blockchain" element={<BlockchainEnhancedDashboard />} />
                      <Route path="marketplace" element={<Marketplace />} />
                      <Route path="omh-integration" element={<OMHIntegrationPage />} />
                      <Route path="docs" element={<Documentation />} />
                      <Route path="settings" element={<Settings />} />
                      <Route path="profile" element={<Profile />} />
                      <Route path="*" element={<Dashboard />} />
                    </Routes>
                  </>
                } />
                {/* Legacy routes for backward compatibility - with Navigation */}
                <Route path="/dashboard" element={<><Navigation /><Dashboard /></>} />
                <Route path="/devices" element={<><Navigation /><DeviceManager /></>} />
                <Route path="/tasks" element={<><Navigation /><TaskExecution /></>} />
                <Route path="/ai-training" element={<><Navigation /><AITraining /></>} />
                <Route path="/analytics" element={<><Navigation /><Analytics /></>} />
                <Route path="/blockchain" element={<><Navigation /><BlockchainEnhancedDashboard /></>} />
                <Route path="/marketplace" element={<><Navigation /><Marketplace /></>} />
                <Route path="/omh-integration" element={<><Navigation /><OMHIntegrationPage /></>} />
                <Route path="/docs" element={<><Navigation /><Documentation /></>} />
                <Route path="/settings" element={<><Navigation /><Settings /></>} />
                <Route path="/profile" element={<><Navigation /><Profile /></>} />
              </Routes>
            </div>
          </OMHAuthProvider>
        </Web3Provider>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App;

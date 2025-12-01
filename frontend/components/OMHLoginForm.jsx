/**
 * OMH Login Form Component
 * OAuth login form for Open Mobile Hub authentication
 */

import React, { useState } from 'react';
import {
  Box,
  Button,
  TextField,
  Typography,
  Paper,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem
} from '@mui/material';
import { useOMHAuth } from '../contexts/OMHAuthContext';

const MOCK_USERS = [
  { username: 'alice.smith', label: 'Alice Smith (User, Tester)', roles: ['user', 'tester'] },
  { username: 'bob.jones', label: 'Bob Jones (Admin)', roles: ['admin'] },
  { username: 'carol.wu', label: 'Carol Wu (User, Inactive)', roles: ['user'], inactive: true },
  { username: 'testuser', label: 'Test User (Beta Tester)', roles: ['user', 'beta_tester'] }
];

const OMHLoginForm = ({ onLoginSuccess }) => {
  const { login, loading, error } = useOMHAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('test123');
  const [localError, setLocalError] = useState('');

  const handleQuickLogin = (mockUsername) => {
    setUsername(mockUsername);
    setPassword('test123');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLocalError('');

    if (!username || !password) {
      setLocalError('Please enter username and password');
      return;
    }

    const result = await login(username, password);
    
    if (result.success) {
      if (onLoginSuccess) {
        onLoginSuccess(result.user);
      }
    } else {
      setLocalError(result.error);
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 4, maxWidth: 500, mx: 'auto', mt: 4 }}>
      <Typography variant="h4" gutterBottom align="center" sx={{ mb: 3 }}>
        🔐 OMH Login
      </Typography>
      
      <Typography variant="body2" color="text.secondary" align="center" sx={{ mb: 3 }}>
        Login with Open Mobile Hub OAuth
      </Typography>

      {(error || localError) && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {localError || error}
        </Alert>
      )}

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2" sx={{ fontWeight: 'bold', mb: 1 }}>
          Quick Login (Mock Users)
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          {MOCK_USERS.map(user => (
            <Button
              key={user.username}
              size="small"
              variant="outlined"
              onClick={() => handleQuickLogin(user.username)}
              disabled={loading}
              sx={{ justifyContent: 'flex-start', textAlign: 'left' }}
            >
              {user.label}
              {user.inactive && ' ⚠️'}
            </Button>
          ))}
        </Box>
      </Alert>

      <form onSubmit={handleSubmit}>
        <TextField
          fullWidth
          label="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          margin="normal"
          disabled={loading}
          required
        />
        
        <TextField
          fullWidth
          label="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          margin="normal"
          disabled={loading}
          required
          helperText="Any password works in mock mode"
        />

        <Button
          fullWidth
          type="submit"
          variant="contained"
          color="primary"
          size="large"
          disabled={loading}
          sx={{ mt: 3 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Login with OMH'}
        </Button>
      </form>

      <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
        <Typography variant="caption" color="text.secondary">
          <strong>Note:</strong> Using OMH Mock Server for testing. 
          In production, this would use real OAuth2 authentication.
        </Typography>
      </Box>
    </Paper>
  );
};

export default OMHLoginForm;


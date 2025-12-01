/**
 * Open Mobile Hub (OMH) Integration Panel
 * 
 * Provides UI for OMH authentication and maps features integrated with AutoRL.
 */
/* eslint-disable no-undef */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Button,
  TextField,
  Typography,
  Box,
  Chip,
  Alert,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  IconButton,
  CircularProgress
} from '@mui/material';
import {
  LocationOn,
  Map,
  Login,
  Logout,
  Refresh,
  CheckCircle,
  Error as ErrorIcon,
  Info
} from '@mui/icons-material';

const API_BASE_URL = (typeof process !== 'undefined' && process.env.REACT_APP_API_URL) || 'http://localhost:8000';

const OMHIntegration = () => {
  const [omhEnabled, setOmhEnabled] = useState(false);
  const [authStatus, setAuthStatus] = useState({ authenticated: false, user: null });
  const [mapsEnabled] = useState(true);
  const [location, setLocation] = useState({ lat: 37.7749, lng: -122.4194 });
  const [address, setAddress] = useState('');
  const [geocodeResult, setGeocodeResult] = useState(null);
  const [nearbyPlaces, setNearbyPlaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkOMHStatus();
  }, []);

  const checkOMHStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/omh/status`);
      const data = await response.json();
      setStatus(data);
      setOmhEnabled(data.auth.enabled || data.maps.enabled);
    } catch (err) {
      console.error('Failed to check OMH status:', err);
      setError('Failed to connect to OMH services');
    }
  };

  const handleLogin = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Get auth URL
      const response = await fetch(
        `${API_BASE_URL}/api/v1/omh/auth/url?redirect_uri=${window.location.origin}/auth/callback`
      );
      const data = await response.json();
      
      if (data.mock_mode) {
        // In mock mode, simulate login
        setAuthStatus({
          authenticated: true,
          user: {
            username: 'demo_user',
            email: 'demo@autorl.app',
            user_id: 'demo_user_123',
            mock_mode: true
          }
        });
        setError(null);
      } else {
        // Redirect to actual auth URL
        window.location.href = data.auth_url;
      }
    } catch (err) {
      setError('Login failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    setAuthStatus({ authenticated: false, user: null });
    setError(null);
  };

  const handleGeocode = async () => {
    if (!address.trim()) {
      setError('Please enter an address');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/omh/maps/geocode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address })
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setGeocodeResult(data);
        setLocation({ lat: data.latitude, lng: data.longitude });
        setError(null);
      }
    } catch (err) {
      setError('Geocoding failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFindNearby = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${API_BASE_URL}/api/v1/omh/maps/nearby`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: location.lat,
          longitude: location.lng,
          radius: 1000,
          place_type: null
        })
      });
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
      } else {
        setNearbyPlaces(data.places || []);
        setError(null);
      }
    } catch (err) {
      setError('Failed to find nearby places: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 1200, margin: '0 auto', p: 2 }}>
      <Card sx={{ mb: 2 }}>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Map color="primary" />
              <Typography variant="h5">Open Mobile Hub Integration</Typography>
            </Box>
          }
          action={
            <FormControlLabel
              control={
                <Switch
                  checked={omhEnabled}
                  onChange={(e) => setOmhEnabled(e.target.checked)}
                  color="primary"
                />
              }
              label="Enable OMH"
            />
          }
        />
        <CardContent>
          {status && (
            <Alert 
              severity={status.auth.enabled || status.maps.enabled ? "success" : "info"}
              icon={<Info />}
              sx={{ mb: 2 }}
            >
              {status.message}
              {status.auth.mock_mode && ' (Auth: Mock Mode)'}
              {status.maps.mock_mode && ' (Maps: Mock Mode)'}
            </Alert>
          )}

          {error && (
            <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
        </CardContent>
      </Card>

      {omhEnabled && (
        <>
          {/* Authentication Section */}
          <Card sx={{ mb: 2 }}>
            <CardHeader
              title={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Login />
                  <Typography variant="h6">Authentication</Typography>
                </Box>
              }
            />
            <CardContent>
              {!authStatus.authenticated ? (
                <Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Login with Open Mobile Hub to secure your automation workflows
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={loading ? <CircularProgress size={20} /> : <Login />}
                    onClick={handleLogin}
                    disabled={loading}
                  >
                    {loading ? 'Connecting...' : 'Login with OMH'}
                  </Button>
                </Box>
              ) : (
                <Box>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                    <CheckCircle color="success" />
                    <Typography variant="body1">
                      Logged in as: <strong>{authStatus.user?.username}</strong>
                    </Typography>
                    {authStatus.user?.mock_mode && (
                      <Chip label="Demo Mode" size="small" color="info" />
                    )}
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                    Email: {authStatus.user?.email}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    User ID: {authStatus.user?.user_id}
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<Logout />}
                    onClick={handleLogout}
                    color="secondary"
                  >
                    Logout
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>

          {/* Maps Section */}
          {mapsEnabled && (
            <Card>
              <CardHeader
                title={
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <LocationOn />
                    <Typography variant="h6">Location Services</Typography>
                  </Box>
                }
              />
              <CardContent>
                {/* Geocoding */}
                <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Geocoding
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Enter address"
                    placeholder="e.g., San Francisco, CA"
                    value={address}
                    onChange={(e) => setAddress(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleGeocode()}
                  />
                  <Button
                    variant="contained"
                    onClick={handleGeocode}
                    disabled={loading || !address.trim()}
                  >
                    Geocode
                  </Button>
                </Box>

                {geocodeResult && (
                  <Alert severity="success" sx={{ mb: 3 }}>
                    <Typography variant="body2">
                      <strong>Location Found:</strong> {geocodeResult.formatted_address}
                    </Typography>
                    <Typography variant="body2">
                      Coordinates: {geocodeResult.latitude.toFixed(4)}, {geocodeResult.longitude.toFixed(4)}
                    </Typography>
                    {geocodeResult.mock_mode && (
                      <Chip label="Mock Data" size="small" sx={{ mt: 1 }} />
                    )}
                  </Alert>
                )}

                <Divider sx={{ my: 2 }} />

                {/* Current Location */}
                <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Current Location
                </Typography>
                <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                  <TextField
                    size="small"
                    label="Latitude"
                    type="number"
                    value={location.lat}
                    onChange={(e) => setLocation({ ...location, lat: parseFloat(e.target.value) })}
                    sx={{ flex: 1 }}
                  />
                  <TextField
                    size="small"
                    label="Longitude"
                    type="number"
                    value={location.lng}
                    onChange={(e) => setLocation({ ...location, lng: parseFloat(e.target.value) })}
                    sx={{ flex: 1 }}
                  />
                </Box>

                <Button
                  variant="contained"
                  startIcon={<LocationOn />}
                  onClick={handleFindNearby}
                  disabled={loading}
                  sx={{ mb: 2 }}
                >
                  Find Nearby Places
                </Button>

                {nearbyPlaces.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      Found {nearbyPlaces.length} places nearby:
                    </Typography>
                    <List dense>
                      {nearbyPlaces.map((place, index) => (
                        <ListItem key={index} divider>
                          <ListItemText
                            primary={place.name}
                            secondary={`${place.type} • ${place.distance_meters}m away • ⭐ ${place.rating}`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}

                <Divider sx={{ my: 2 }} />

                {/* Location-Based Automation */}
                <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
                  Location-Based Automation
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  Create workflows triggered by location (geofencing):
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip
                    label="🏠 Trigger when arriving home"
                    onClick={() => console.log('Home geofence')}
                    variant="outlined"
                  />
                  <Chip
                    label="🏢 Trigger when leaving office"
                    onClick={() => console.log('Office geofence')}
                    variant="outlined"
                  />
                  <Chip
                    label="🛍️ Trigger near store"
                    onClick={() => console.log('Store geofence')}
                    variant="outlined"
                  />
                </Box>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  );
};

export default OMHIntegration;


/**
 * OMH Integration Dashboard Page
 * Shows OMH authentication status, user profile, and location-aware features
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  Grid,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  Chip,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import {
  Person as PersonIcon,
  LocationOn as LocationIcon,
  VpnKey as KeyIcon,
  Logout as LogoutIcon,
  Task as TaskIcon,
  Refresh as RefreshIcon,
  Map as MapIcon
} from '@mui/icons-material';
import { useOMHAuth } from '../contexts/OMHAuthContext';
import OMHLoginForm from '../components/OMHLoginForm';

const OMHIntegrationPage = () => {
  const {
    user,
    accessToken,
    location,
    isAuthenticated,
    logout,
    fetchUserLocation,
    getNearbyTasks,
    getAutoRLProfile
  } = useOMHAuth();

  const [nearbyTasks, setNearbyTasks] = useState([]);
  const [autorlProfile, setAutoRLProfile] = useState(null);
  const [loading, setLoading] = useState(false);

  const loadNearbyTasks = useCallback(async () => {
    try {
      const data = await getNearbyTasks();
      setNearbyTasks(data.tasks || []);
    } catch (err) {
      console.error('Failed to load nearby tasks:', err);
    }
  }, [getNearbyTasks]);

  const loadAutoRLProfile = useCallback(async () => {
    try {
      const data = await getAutoRLProfile();
      setAutoRLProfile(data);
    } catch (err) {
      console.error('Failed to load AutoRL profile:', err);
    }
  }, [getAutoRLProfile]);

  useEffect(() => {
    if (isAuthenticated) {
      loadNearbyTasks();
      loadAutoRLProfile();
    }
  }, [isAuthenticated, loadNearbyTasks, loadAutoRLProfile]);

  const handleRefreshLocation = async () => {
    setLoading(true);
    await fetchUserLocation();
    await loadNearbyTasks();
    setLoading(false);
  };

  if (!isAuthenticated) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <OMHLoginForm onLoginSuccess={() => {
          loadNearbyTasks();
          loadAutoRLProfile();
        }} />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h3" sx={{ fontWeight: 'bold' }}>
          OMH Integration Dashboard
        </Typography>
        <Button
          variant="outlined"
          color="error"
          startIcon={<LogoutIcon />}
          onClick={logout}
        >
          Logout
        </Button>
      </Box>

      <Grid container spacing={3}>
        {/* User Profile Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <PersonIcon sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="h5">User Profile</Typography>
              </Box>
              
              <Divider sx={{ mb: 2 }} />
              
              <List dense>
                <ListItem>
                  <ListItemText 
                    primary="Username" 
                    secondary={user?.username} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Full Name" 
                    secondary={user?.full_name} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Email" 
                    secondary={user?.email} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="User ID" 
                    secondary={user?.user_id} 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Account Status" 
                    secondary={
                      <Chip 
                        label={user?.account_status} 
                        size="small" 
                        color={user?.account_status === 'active' ? 'success' : 'default'}
                      />
                    } 
                  />
                </ListItem>
                <ListItem>
                  <ListItemText 
                    primary="Roles" 
                    secondary={
                      <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                        {user?.roles?.map(role => (
                          <Chip key={role} label={role} size="small" color="primary" />
                        ))}
                      </Box>
                    } 
                  />
                </ListItem>
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* Location Card */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <LocationIcon sx={{ mr: 1, color: 'secondary.main' }} />
                  <Typography variant="h5">Location Context</Typography>
                </Box>
                <Button
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={handleRefreshLocation}
                  disabled={loading}
                >
                  Refresh
                </Button>
              </Box>
              
              <Divider sx={{ mb: 2 }} />
              
              {location ? (
                <List dense>
                  <ListItem>
                    <ListItemText 
                      primary="Address" 
                      secondary={location.address} 
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="Coordinates" 
                      secondary={`${location.latitude}, ${location.longitude}`} 
                    />
                  </ListItem>
                  <ListItem>
                    <ListItemText 
                      primary="Place ID" 
                      secondary={location.place_id} 
                    />
                  </ListItem>
                  <ListItem>
                    <Button
                      fullWidth
                      variant="outlined"
                      startIcon={<MapIcon />}
                      href={`https://www.google.com/maps?q=${location.latitude},${location.longitude}`}
                      target="_blank"
                    >
                      View on Map
                    </Button>
                  </ListItem>
                </List>
              ) : (
                <Alert severity="info">Location not available</Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Access Token Card */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <KeyIcon sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="h5">OAuth Access Token</Typography>
              </Box>
              
              <Divider sx={{ mb: 2 }} />
              
              <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default', fontFamily: 'monospace', overflow: 'auto' }}>
                <Typography variant="caption" sx={{ wordBreak: 'break-all' }}>
                  {accessToken}
                </Typography>
              </Paper>
              
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                This token is used for authenticated requests to AutoRL API
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Nearby Tasks Card */}
        {location && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TaskIcon sx={{ mr: 1, color: 'success.main' }} />
                  <Typography variant="h5">Nearby Location-Aware Tasks</Typography>
                </Box>
                
                <Divider sx={{ mb: 2 }} />
                
                {nearbyTasks.length > 0 ? (
                  <List>
                    {nearbyTasks.map((task, index) => (
                      <ListItem key={index} divider>
                        <ListItemIcon>
                          <TaskIcon color="primary" />
                        </ListItemIcon>
                        <ListItemText 
                          primary={task.name}
                          secondary={`Type: ${task.type} • Distance: ${task.distance}`}
                        />
                      </ListItem>
                    ))}
                  </List>
                ) : (
                  <Alert severity="info">No nearby tasks available</Alert>
                )}
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* AutoRL Integration Status */}
        {autorlProfile && (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography variant="h5" gutterBottom>
                  AutoRL Integration Status
                </Typography>
                
                <Divider sx={{ my: 2 }} />
                
                <Alert severity="success" sx={{ mb: 2 }}>
                  Successfully connected to AutoRL API with OMH authentication
                </Alert>
                
                <Typography variant="body2" color="text.secondary">
                  <strong>Auth Method:</strong> {autorlProfile.auth_method}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>Profile:</strong> {autorlProfile.profile?.full_name} ({autorlProfile.profile?.username})
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>
    </Container>
  );
};

export default OMHIntegrationPage;


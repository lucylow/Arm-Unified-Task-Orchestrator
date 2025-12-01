/**
 * OMH Authentication Context
 * Provides OAuth authentication via Open Mobile Hub for AutoRL
 */

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

const OMHAuthContext = createContext(null);

const OMH_API_BASE = import.meta.env.VITE_OMH_API_BASE || 'http://localhost:8001';
const AUTORL_API_BASE = import.meta.env.VITE_AUTORL_API_BASE || 'http://localhost:8000';

export const OMHAuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [accessToken, setAccessToken] = useState(null);
  const [location, setLocation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  /**
   * Fetch user's current location from OMH
   */
  const fetchUserLocation = useCallback(async (token = accessToken) => {
    if (!token) return;

    try {
      const response = await fetch(`${OMH_API_BASE}/api/v1/maps/location`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (response.ok) {
        const locationData = await response.json();
        setLocation(locationData.location);
      }
    } catch (err) {
      console.error('Failed to fetch location:', err);
    }
  }, [accessToken]);

  // Load token from localStorage on mount
  useEffect(() => {
    const storedToken = localStorage.getItem('omh_access_token');
    const storedUser = localStorage.getItem('omh_user');
    
    if (storedToken && storedUser) {
      setAccessToken(storedToken);
      setUser(JSON.parse(storedUser));
      fetchUserLocation(storedToken);
    }
    
    setLoading(false);
  }, [fetchUserLocation]);

  /**
   * Login with OMH OAuth
   */
  const login = async (username, password) => {
    setLoading(true);
    setError(null);

    try {
      // Authenticate with OMH Mock Server
      const authResponse = await fetch(`${OMH_API_BASE}/auth/token`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          grant_type: 'password',
          username,
          password
        })
      });

      if (!authResponse.ok) {
        throw new Error('Authentication failed');
      }

      const authData = await authResponse.json();
      const token = authData.access_token;

      // Get user profile
      const profileResponse = await fetch(`${OMH_API_BASE}/api/v1/user/profile`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (!profileResponse.ok) {
        throw new Error('Failed to fetch user profile');
      }

      const userProfile = await profileResponse.json();

      // Store in state and localStorage
      setAccessToken(token);
      setUser(userProfile);
      localStorage.setItem('omh_access_token', token);
      localStorage.setItem('omh_user', JSON.stringify(userProfile));

      // Fetch location
      await fetchUserLocation(token);

      setLoading(false);
      return { success: true, user: userProfile };
    } catch (err) {
      setError(err.message);
      setLoading(false);
      return { success: false, error: err.message };
    }
  };

  /**
   * Logout user
   */
  const logout = () => {
    setUser(null);
    setAccessToken(null);
    setLocation(null);
    localStorage.removeItem('omh_access_token');
    localStorage.removeItem('omh_user');
  };

  /**
   * Execute AutoRL task with location context
   */
  const executeLocationAwareTask = async (instruction, deviceId, parameters = {}) => {
    if (!accessToken) {
      throw new Error('Not authenticated');
    }

    try {
      const response = await fetch(`${AUTORL_API_BASE}/api/v1/execute/location-aware`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`
        },
        body: JSON.stringify({
          instruction,
          device_id: deviceId,
          parameters
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Task execution failed');
      }

      return await response.json();
    } catch (err) {
      throw new Error(err.message);
    }
  };

  /**
   * Get nearby tasks based on user location
   */
  const getNearbyTasks = async () => {
    if (!accessToken) {
      throw new Error('Not authenticated');
    }

    try {
      const response = await fetch(`${AUTORL_API_BASE}/api/v1/location/nearby-tasks`, {
        headers: { 'Authorization': `Bearer ${accessToken}` }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch nearby tasks');
      }

      return await response.json();
    } catch (err) {
      throw new Error(err.message);
    }
  };

  /**
   * Get user profile via AutoRL API (with OMH auth)
   */
  const getAutoRLProfile = async () => {
    if (!accessToken) {
      throw new Error('Not authenticated');
    }

    try {
      const response = await fetch(`${AUTORL_API_BASE}/api/v1/user/profile/omh`, {
        headers: { 'Authorization': `Bearer ${accessToken}` }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch AutoRL profile');
      }

      return await response.json();
    } catch (err) {
      throw new Error(err.message);
    }
  };

  const value = {
    user,
    accessToken,
    location,
    loading,
    error,
    isAuthenticated: !!accessToken,
    login,
    logout,
    fetchUserLocation,
    executeLocationAwareTask,
    getNearbyTasks,
    getAutoRLProfile
  };

  return (
    <OMHAuthContext.Provider value={value}>
      {children}
    </OMHAuthContext.Provider>
  );
};

/**
 * Hook to use OMH authentication context
 */
// eslint-disable-next-line react-refresh/only-export-components
export const useOMHAuth = () => {
  const context = useContext(OMHAuthContext);
  if (!context) {
    throw new Error('useOMHAuth must be used within OMHAuthProvider');
  }
  return context;
};

export default OMHAuthContext;


import React, { useState } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  Button, 
  Avatar, 
  Chip, 
  IconButton, 
  LinearProgress,
  Badge,
  Tooltip,
  useTheme,
  alpha,
} from '@mui/material';
import { 
  Devices, 
  Android, 
  Apple, 
  Battery90, 
  Battery30,
  BatteryFull,
  Wifi,
  SignalCellular4Bar,
  Lock,
  Refresh,
  PlayArrow,
  Settings,
  Add,
  Circle,
} from '@mui/icons-material';

const DeviceManager = () => {
  const theme = useTheme();
  const [devices] = useState([
    { 
      id: 'android_pixel_7_1', 
      name: 'Pixel 7', 
      platform: 'Android', 
      status: 'active', 
      battery: 87, 
      connection: 'WiFi',
      tasks: 12,
      uptime: '4h 23m',
      model: 'Pixel 7 Pro',
      os: 'Android 14',
    },
    { 
      id: 'iphone_14_1', 
      name: 'iPhone 14', 
      platform: 'iOS', 
      status: 'active', 
      battery: 65, 
      connection: 'Cellular',
      tasks: 8,
      uptime: '2h 15m',
      model: 'iPhone 14 Pro',
      os: 'iOS 17.2',
    },
    { 
      id: 'android_galaxy_s23', 
      name: 'Galaxy S23', 
      platform: 'Android', 
      status: 'busy', 
      battery: 92, 
      connection: 'WiFi',
      tasks: 15,
      uptime: '6h 47m',
      model: 'Galaxy S23 Ultra',
      os: 'Android 14',
    },
    { 
      id: 'android_emulator_1', 
      name: 'Android Emulator', 
      platform: 'Android', 
      status: 'offline', 
      battery: 100, 
      connection: 'Ethernet',
      tasks: 0,
      uptime: '0m',
      model: 'Emulator',
      os: 'Android 13',
    },
    { 
      id: 'iphone_13_1', 
      name: 'iPhone 13', 
      platform: 'iOS', 
      status: 'active', 
      battery: 78, 
      connection: 'WiFi',
      tasks: 5,
      uptime: '1h 42m',
      model: 'iPhone 13',
      os: 'iOS 17.1',
    },
    { 
      id: 'android_pixel_6a', 
      name: 'Pixel 6a', 
      platform: 'Android', 
      status: 'idle', 
      battery: 45, 
      connection: 'WiFi',
      tasks: 3,
      uptime: '5h 12m',
      model: 'Pixel 6a',
      os: 'Android 14',
    },
  ]);

  const [filter, setFilter] = useState('all');

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return theme.palette.info.main;
      case 'busy':
        return theme.palette.primary.main;
      case 'idle':
        return theme.palette.secondary.main;
      case 'offline':
        return theme.palette.text.disabled;
      default:
        return theme.palette.text.secondary;
    }
  };

  const getBatteryIcon = (battery) => {
    if (battery > 80) return <BatteryFull />;
    if (battery > 30) return <Battery90 />;
    return <Battery30 />;
  };

  const getPlatformIcon = (platform) => {
    return platform === 'iOS' ? <Apple /> : <Android />;
  };

  const filteredDevices = filter === 'all' 
    ? devices 
    : devices.filter(d => d.status === filter);

  return (
    <Box sx={{ pt: 10, pb: 4, minHeight: '100vh' }}>
      <Container maxWidth="xl">
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 4 }}>
          <Box>
            <Typography 
              variant="h3" 
              fontWeight="700" 
              sx={{
                background: 'linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 1,
              }}
            >
          Device Manager
        </Typography>
            <Typography variant="body1" color="text.secondary">
              Monitor and control connected devices
            </Typography>
          </Box>
          <Button 
            variant="contained" 
            startIcon={<Add />}
            sx={{
              px: 3,
              py: 1.5,
              background: 'linear-gradient(135deg, #1e88e5 0%, #1565c0 100%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%)',
                transform: 'scale(1.05)',
              },
              transition: 'all 0.3s',
            }}
          >
            Add Device
          </Button>
        </Box>

        {/* Filter Chips */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          {['all', 'active', 'busy', 'idle', 'offline'].map((status) => (
            <Chip
              key={status}
              label={status.charAt(0).toUpperCase() + status.slice(1)}
              onClick={() => setFilter(status)}
              sx={{
                px: 2,
                py: 2.5,
                fontSize: '0.9rem',
                fontWeight: 600,
                textTransform: 'capitalize',
                bgcolor: filter === status ? alpha(theme.palette.primary.main, 0.2) : alpha(theme.palette.background.paper, 0.5),
                color: filter === status ? theme.palette.primary.main : theme.palette.text.secondary,
                border: `2px solid ${filter === status ? theme.palette.primary.main : 'transparent'}`,
                transition: 'all 0.3s',
                '&:hover': {
                  bgcolor: alpha(theme.palette.primary.main, 0.15),
                  transform: 'scale(1.05)',
                },
              }}
            />
          ))}
        </Box>

        {/* Device Grid */}
        <Grid container spacing={3}>
          {filteredDevices.map((device) => (
            <Grid item xs={12} sm={6} lg={4} key={device.id}>
              <Card 
                sx={{ 
                  height: '100%',
                  position: 'relative',
                  overflow: 'visible',
                  transition: 'all 0.3s',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: `0 12px 24px ${alpha(getStatusColor(device.status), 0.3)}`,
                  }
                }}
              >
                {/* Status Ring */}
                <Box
                  sx={{
                    position: 'absolute',
                    top: -8,
                    right: -8,
                    width: 24,
                    height: 24,
                    borderRadius: '50%',
                    bgcolor: getStatusColor(device.status),
                    boxShadow: `0 0 20px ${getStatusColor(device.status)}`,
                    animation: device.status === 'active' ? 'pulse-blue 2s infinite' : 'none',
                    zIndex: 1,
                  }}
                />
                
                <CardContent sx={{ p: 3 }}>
                  {/* Device Header */}
                  <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', mb: 3 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Badge
                        overlap="circular"
                        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
                        badgeContent={
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              bgcolor: getStatusColor(device.status),
                              border: `2px solid ${theme.palette.background.paper}`,
                            }}
                          />
                        }
                      >
                        <Avatar 
                          sx={{ 
                            width: 56, 
                            height: 56,
                            bgcolor: alpha(theme.palette.primary.main, 0.1),
                            color: theme.palette.primary.main,
                          }}
                        >
                          {getPlatformIcon(device.platform)}
                  </Avatar>
                      </Badge>
                      <Box>
                        <Typography variant="h6" fontWeight="600">
                          {device.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {device.model}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>

                  {/* Device Info */}
                  <Box sx={{ mb: 3 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Battery
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        {getBatteryIcon(device.battery)}
                        <Typography variant="body2" fontWeight="600" color="primary">
                          {device.battery}%
                        </Typography>
                      </Box>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={device.battery} 
                      sx={{ 
                        height: 6, 
                        borderRadius: 3,
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        '& .MuiLinearProgress-bar': {
                          bgcolor: device.battery > 30 ? theme.palette.info.main : theme.palette.error.main,
                          borderRadius: 3,
                        }
                      }} 
                    />
                  </Box>

                  {/* Stats Grid */}
                  <Grid container spacing={2} sx={{ mb: 3 }}>
                    <Grid item xs={6}>
                      <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: alpha(theme.palette.primary.main, 0.05), borderRadius: 2 }}>
                        <Typography variant="h5" fontWeight="700" color="primary">
                          {device.tasks}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Tasks
                        </Typography>
                      </Box>
                    </Grid>
                    <Grid item xs={6}>
                      <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: alpha(theme.palette.info.main, 0.05), borderRadius: 2 }}>
                        <Typography variant="h5" fontWeight="700" color="info.main">
                          {device.uptime}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Uptime
                        </Typography>
                      </Box>
                    </Grid>
                  </Grid>

                  {/* Device Details */}
                  <Box sx={{ mb: 3 }}>
                    <Chip 
                      label={device.platform} 
                      size="small" 
                      sx={{ 
                        mr: 1,
                        bgcolor: alpha(theme.palette.primary.main, 0.1),
                        color: theme.palette.primary.main,
                      }} 
                    />
                    <Chip 
                      label={device.os} 
                      size="small" 
                      sx={{ 
                        mr: 1,
                        bgcolor: alpha(theme.palette.secondary.main, 0.1),
                        color: theme.palette.secondary.main,
                      }} 
                    />
                    <Chip 
                      label={device.connection} 
                      size="small" 
                      icon={device.connection === 'WiFi' ? <Wifi /> : <SignalCellular4Bar />}
                      sx={{ 
                        bgcolor: alpha(theme.palette.info.main, 0.1),
                        color: theme.palette.info.main,
                      }} 
                    />
                  </Box>

                  {/* Status Badge */}
                  <Chip 
                    label={device.status.toUpperCase()} 
                    size="small" 
                    sx={{ 
                      mb: 3,
                      width: '100%',
                      fontWeight: 700,
                      bgcolor: alpha(getStatusColor(device.status), 0.15),
                      color: getStatusColor(device.status),
                      border: `2px solid ${getStatusColor(device.status)}`,
                    }} 
                  />

                  {/* Action Buttons */}
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Tooltip title="Assign Task">
                      <IconButton 
                        size="small"
                        disabled={device.status === 'offline'}
                        sx={{ 
                          flex: 1,
                          bgcolor: alpha(theme.palette.primary.main, 0.1),
                          '&:hover': {
                            bgcolor: alpha(theme.palette.primary.main, 0.2),
                          },
                        }}
                      >
                        <PlayArrow />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Refresh">
                      <IconButton 
                        size="small"
                        sx={{ 
                          flex: 1,
                          bgcolor: alpha(theme.palette.info.main, 0.1),
                          '&:hover': {
                            bgcolor: alpha(theme.palette.info.main, 0.2),
                          },
                        }}
                      >
                        <Refresh />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Settings">
                      <IconButton 
                        size="small"
                        sx={{ 
                          flex: 1,
                          bgcolor: alpha(theme.palette.secondary.main, 0.1),
                          '&:hover': {
                            bgcolor: alpha(theme.palette.secondary.main, 0.2),
                          },
                        }}
                      >
                        <Settings />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Lock">
                      <IconButton 
                        size="small"
                        disabled={device.status === 'offline'}
                        sx={{ 
                          flex: 1,
                          bgcolor: alpha(theme.palette.text.secondary, 0.1),
                          '&:hover': {
                            bgcolor: alpha(theme.palette.text.secondary, 0.2),
                          },
                        }}
                      >
                        <Lock />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
};

export default DeviceManager;

import React, { useState, useMemo } from 'react';
import {
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  AppBar,
  Toolbar,
  Typography,
  Box,
  Breadcrumbs,
  Link,
  Tooltip,
  Chip,
  useTheme
} from '@mui/material';
import {
  Home,
  Dashboard,
  Devices,
  PlayArrow,
  Psychology,
  TrendingUp,
  Store,
  Description,
  Settings,
  AccountCircle,
  Menu as MenuIcon,
  Circle as CircleIcon
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const menuItems = [
  { title: 'Home', path: '/', icon: <Home />, description: 'Landing page and overview' },
  { title: 'Dashboard', path: '/app/dashboard', icon: <Dashboard />, description: 'Real-time system monitoring' },
  { title: 'Device Manager', path: '/app/devices', icon: <Devices />, description: 'Manage Android/iOS devices' },
  { title: 'Task Execution', path: '/app/tasks', icon: <PlayArrow />, description: 'Create and monitor automation tasks' },
  { title: 'AI Training', path: '/app/ai-training', icon: <Psychology />, description: 'RL training and model management' },
  { title: 'Analytics', path: '/app/analytics', icon: <TrendingUp />, description: 'Performance metrics and insights' },
  { title: 'Marketplace', path: '/app/marketplace', icon: <Store />, description: 'Workflow plugins and extensions' },
  { title: 'Documentation', path: '/app/docs', icon: <Description />, description: 'API docs and guides' },
  { title: 'Settings', path: '/app/settings', icon: <Settings />, description: 'System configuration' },
  { title: 'Profile', path: '/app/profile', icon: <AccountCircle />, description: 'User account' },
];

const StatusIndicator = ({ label = 'OK', color = 'success' }) => (
  <Tooltip title={`System status: ${label}`}>
    <Chip label={label} color={color} size="small" sx={{ ml: 1 }} />
  </Tooltip>
);

const BreadcrumbsNav = ({ location }) => {
  const pathnames = location.pathname.split('/').filter((x) => x);
  return (
    <Breadcrumbs aria-label="breadcrumb" sx={{ color: 'text.secondary' }}>
      <Link underline="hover" color="inherit" href="#" onClick={(e) => e.preventDefault()} aria-current={pathnames.length === 0 ? 'page' : undefined}>
        Home
      </Link>
      {pathnames.map((value, index) => {
        const to = `/${pathnames.slice(0, index + 1).join('/')}`;
        const isLast = index === pathnames.length - 1;
        return (
          <Typography key={to} color={isLast ? 'text.primary' : 'text.secondary'} sx={{ fontSize: 13 }}>
            {value.replace(/-/g, ' ')}
          </Typography>
        );
      })}
    </Breadcrumbs>
  );
};

const Navigation = () => {
  const [drawerOpen, setDrawerOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();

  const handleDrawerToggle = () => setDrawerOpen(!drawerOpen);
  const handleNavigation = (path) => {
    navigate(path);
    setDrawerOpen(false);
  };

  const isActive = (path) => location.pathname === path;

  const status = useMemo(() => ({ label: 'Connected', color: 'info' }), []);

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Top Navigation Bar with Blue Gradient */}
      <AppBar 
        position="sticky" 
        elevation={0}
        sx={{
          background: 'linear-gradient(135deg, rgba(30, 136, 229, 0.95) 0%, rgba(21, 101, 192, 0.95) 100%)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(66, 165, 245, 0.2)',
          boxShadow: '0 4px 20px rgba(30, 136, 229, 0.3)',
        }}
      >
        <Toolbar sx={{ display: 'flex', justifyContent: 'space-between', gap: 2, minHeight: 64 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton 
              edge="start" 
              color="inherit" 
              aria-label="menu" 
              onClick={handleDrawerToggle} 
              sx={{ 
                mr: 1,
                transition: 'all 0.3s',
                '&:hover': {
                  transform: 'scale(1.1)',
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                }
              }}
            >
              <MenuIcon />
            </IconButton>
            <Typography 
              variant="h6" 
              sx={{ 
                fontWeight: 700, 
                letterSpacing: 1,
                background: 'linear-gradient(90deg, #ffffff 0%, #e3f2fd 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              AutoRL Platform
            </Typography>
            <Box sx={{ ml: 2 }}>
              <StatusIndicator label={status.label} color={status.color} />
            </Box>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <BreadcrumbsNav location={location} />
            <IconButton 
              color="inherit" 
              aria-label="account" 
              onClick={() => navigate('/app/profile')}
              sx={{
                transition: 'all 0.3s',
                '&:hover': {
                  transform: 'scale(1.1)',
                  bgcolor: 'rgba(255, 255, 255, 0.1)',
                }
              }}
            >
              <AccountCircle />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Glassmorphic Sidebar Drawer */}
      <Drawer 
        anchor="left" 
        open={drawerOpen} 
        onClose={handleDrawerToggle}
        PaperProps={{
          sx: {
            width: 280,
            background: 'linear-gradient(180deg, rgba(18, 24, 43, 0.95) 0%, rgba(13, 17, 31, 0.95) 100%)',
            backdropFilter: 'blur(20px)',
            borderRight: '1px solid rgba(30, 136, 229, 0.2)',
            boxShadow: '4px 0 20px rgba(30, 136, 229, 0.2)',
          }
        }}
      >
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }} role="presentation">
          {/* Sidebar Header */}
          <Box 
            sx={{ 
              p: 3, 
              borderBottom: `1px solid rgba(30, 136, 229, 0.2)`,
              background: 'linear-gradient(135deg, rgba(30, 136, 229, 0.1) 0%, rgba(21, 101, 192, 0.1) 100%)',
            }}
          >
            <Typography 
              variant="h5" 
              sx={{ 
                fontWeight: 700,
                background: 'linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                mb: 0.5,
              }}
            >
              AutoRL
            </Typography>
            <Typography variant="body2" sx={{ color: theme.palette.text.secondary }}>
              AI Agent Command Center
            </Typography>
          </Box>

          {/* Navigation Menu */}
          <List sx={{ flex: 1, px: 1, py: 2 }}>
            {menuItems.map((item) => (
              <ListItem 
                button 
                key={item.title} 
                selected={isActive(item.path)} 
                onClick={() => handleNavigation(item.path)}
                sx={{
                  borderRadius: 2,
                  mb: 0.5,
                  transition: 'all 0.3s',
                  position: 'relative',
                  overflow: 'hidden',
                  '&::before': isActive(item.path) ? {
                    content: '""',
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: 4,
                    bgcolor: theme.palette.primary.main,
                    borderRadius: '0 4px 4px 0',
                  } : {},
                  '&.Mui-selected': {
                    bgcolor: 'rgba(30, 136, 229, 0.15)',
                    borderLeft: `4px solid ${theme.palette.primary.main}`,
                    '&:hover': {
                      bgcolor: 'rgba(30, 136, 229, 0.25)',
                    }
                  },
                  '&:hover': {
                    bgcolor: 'rgba(30, 136, 229, 0.1)',
                    transform: 'translateX(4px)',
                  }
                }}
              >
                <ListItemIcon 
                  sx={{ 
                    color: isActive(item.path) ? theme.palette.primary.main : theme.palette.text.secondary,
                    minWidth: 40,
                    transition: 'all 0.3s',
                  }}
                >
                  {item.icon}
                </ListItemIcon>
                <ListItemText 
                  primary={item.title} 
                  secondary={item.description}
                  primaryTypographyProps={{
                    fontWeight: isActive(item.path) ? 600 : 400,
                    color: isActive(item.path) ? theme.palette.primary.light : theme.palette.text.primary,
                  }}
                  secondaryTypographyProps={{
                    fontSize: '0.75rem',
                    color: theme.palette.text.secondary,
                  }}
                />
              </ListItem>
            ))}
          </List>

          {/* Sidebar Footer with Status */}
          <Box 
            sx={{ 
              p: 2, 
              borderTop: `1px solid rgba(30, 136, 229, 0.2)`,
              background: 'linear-gradient(135deg, rgba(30, 136, 229, 0.05) 0%, rgba(21, 101, 192, 0.05) 100%)',
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1, gap: 1 }}>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: '#00d4ff',
                  boxShadow: '0 0 10px #00d4ff',
                  animation: 'pulse-blue 2s infinite',
                }}
              />
              <Typography variant="body2" sx={{ color: theme.palette.info.light, fontWeight: 600 }}>
                WebSocket Connected
              </Typography>
            </Box>
            <Typography variant="caption" sx={{ color: theme.palette.text.secondary, display: 'block', mb: 0.5 }}>
              3 Devices Online
            </Typography>
            <Typography variant="caption" sx={{ color: theme.palette.text.secondary, opacity: 0.7 }}>
              v1.0.0 • Enterprise Edition
            </Typography>
          </Box>
        </Box>
      </Drawer>
    </Box>
  );
};

export default Navigation;

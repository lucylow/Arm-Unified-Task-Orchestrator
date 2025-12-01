import React from 'react';
import { 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  useTheme,
  useMediaQuery 
} from '@mui/material';

const MobileOptimizedCard = ({ title, value, icon, color = 'primary' }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Card 
      elevation={isMobile ? 1 : 3} 
      sx={{ 
        height: '100%',
        minHeight: isMobile ? 120 : 150,
        cursor: 'pointer',
        transition: 'all 0.3s ease',
        '&:hover': {
          transform: isMobile ? 'none' : 'translateY(-4px)',
          boxShadow: theme.shadows[6],
        },
        '&:active': {
          transform: isMobile ? 'scale(0.98)' : 'translateY(-2px)',
        }
      }}
    >
      <CardContent sx={{ 
        display: 'flex', 
        flexDirection: isMobile ? 'row' : 'column',
        alignItems: isMobile ? 'center' : 'flex-start',
        height: '100%',
        p: isMobile ? 2 : 3,
        '&:last-child': { pb: isMobile ? 2 : 3 }
      }}>
        <Box sx={{ 
          mb: isMobile ? 0 : 2, 
          mr: isMobile ? 2 : 0,
          color: `${color}.main` 
        }}>
          {icon}
        </Box>
        <Box sx={{ flexGrow: 1 }}>
          <Typography 
            variant={isMobile ? "body2" : "h6"} 
            component="h2" 
            color="text.secondary"
            gutterBottom={!isMobile}
          >
            {title}
          </Typography>
          <Typography 
            variant={isMobile ? "h6" : "h4"} 
            color={`${color}.main`}
            fontWeight="bold"
          >
            {value}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default MobileOptimizedCard;

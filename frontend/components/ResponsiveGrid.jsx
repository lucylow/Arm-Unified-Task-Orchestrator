import React from 'react';
import { Grid, useTheme, useMediaQuery } from '@mui/material';

const ResponsiveGrid = ({ children, spacing = 3 }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  return (
    <Grid 
      container 
      spacing={isMobile ? 2 : spacing}
      sx={{
        px: isMobile ? 1 : 0,
        py: isMobile ? 1 : 0,
      }}
    >
      {React.Children.map(children, (child, index) => (
        <Grid 
          item 
          xs={12} 
          sm={6} 
          md={4} 
          lg={3}
          key={index}
          sx={{
            '&:last-child': {
              mb: isMobile ? 2 : 0
            }
          }}
        >
          {child}
        </Grid>
      ))}
    </Grid>
  );
};

export default ResponsiveGrid;

import React from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Grid, 
  Card, 
  CardContent, 
  List, 
  ListItem, 
  ListItemText,
  Paper
} from '@mui/material';
import { 
  Description, 
  Code, 
  FlashOn, 
  Security,
  MenuBook 
} from '@mui/icons-material';

const Documentation = () => {
  const sections = [
    {
      title: 'Getting Started',
      icon: FlashOn,
      color: '#1e88e5',
      items: [
        { primary: 'Installation & Setup', secondary: 'Set up AutoRL on your system' },
        { primary: 'Connecting Devices', secondary: 'Link your Android/iOS devices' },
        { primary: 'Creating Your First Task', secondary: 'Build your first automation workflow' },
        { primary: 'Understanding the Dashboard', secondary: 'Navigate the AutoRL interface' }
      ]
    },
    {
      title: 'API Reference',
      icon: Code,
      color: '#0288d1',
      items: [
        { primary: 'POST /api/v1/execute', secondary: 'Execute automation task' },
        { primary: 'GET /api/v1/devices', secondary: 'List connected devices' },
        { primary: 'GET /api/v1/tasks/{id}', secondary: 'Get task status' },
        { primary: 'POST /api/v1/train', secondary: 'Start RL training' }
      ]
    },
    {
      title: 'Best Practices',
      icon: MenuBook,
      color: '#42a5f5',
      items: [
        { primary: 'Writing Effective Task Descriptions', secondary: 'Tips for natural language tasks' },
        { primary: 'Device Management Tips', secondary: 'Optimize your device setup' },
        { primary: 'Optimizing Task Performance', secondary: 'Improve execution speed' },
        { primary: 'Debugging Failed Tasks', secondary: 'Troubleshooting guide' }
      ]
    },
    {
      title: 'Security',
      icon: Security,
      color: '#00b0ff',
      items: [
        { primary: 'PII Data Masking', secondary: 'Protect sensitive information' },
        { primary: 'Safety Guardrails', secondary: 'Built-in security measures' },
        { primary: 'Secure API Access', secondary: 'Authentication and authorization' },
        { primary: 'Privacy Guidelines', secondary: 'Data handling best practices' }
      ]
    }
  ];

  return (
    <Box sx={{ pt: 10, pb: 4 }}>
      <Container maxWidth="xl">
        <Box sx={{ mb: 6 }}>
          <Typography variant="h3" fontWeight="bold" sx={{ mb: 2 }}>
            Documentation
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Guides, API reference, and best practices for AutoRL Platform
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {sections.map((section) => {
            const IconComponent = section.icon;
            return (
              <Grid item xs={12} md={6} key={section.title}>
                <Card sx={{ height: '100%', transition: 'all 0.3s', '&:hover': { transform: 'translateY(-4px)', boxShadow: 4 } }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                      <IconComponent sx={{ color: section.color, mr: 1.5, fontSize: 28 }} />
                      <Typography variant="h6" fontWeight="600">
                        {section.title}
                      </Typography>
                    </Box>
                    <List dense>
                      {section.items.map((item, index) => (
                        <ListItem 
                          key={index}
                          sx={{ 
                            borderRadius: 1,
                            mb: 0.5,
                            transition: 'all 0.2s',
                            '&:hover': { 
                              bgcolor: 'action.hover',
                              cursor: 'pointer',
                              pl: 3
                            }
                          }}
                        >
                          <ListItemText 
                            primary={item.primary} 
                            secondary={item.secondary}
                            primaryTypographyProps={{ fontWeight: 500, fontSize: '0.95rem' }}
                            secondaryTypographyProps={{ fontSize: '0.85rem' }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        <Paper sx={{ mt: 4, p: 4, bgcolor: 'background.paper' }} elevation={2}>
          <Typography variant="h5" fontWeight="600" sx={{ mb: 3 }}>
            Quick Start Example
          </Typography>
          <Paper sx={{ bgcolor: '#1a1a1a', p: 3, borderRadius: 2, border: '1px solid rgba(30, 136, 229, 0.3)' }}>
            <Typography component="pre" sx={{ 
              fontFamily: 'monospace', 
              fontSize: '0.9rem', 
              color: '#42a5f5',
              overflow: 'auto',
              m: 0
            }}>
{`// Example: Execute a task
const response = await fetch('/api/v1/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    device_id: 'android_pixel_7',
    task_description: 'Open Instagram and like the latest post',
    max_steps: 10
  })
});

const result = await response.json();
console.log(result);`}
            </Typography>
          </Paper>
        </Paper>
      </Container>
    </Box>
  );
};

export default Documentation;

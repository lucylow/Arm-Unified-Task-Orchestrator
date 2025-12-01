import React from 'react';
import { Card, CardContent, Typography, LinearProgress, Avatar, Box } from '@mui/material';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import useAgentStatus from '../../hooks/useAgentStatus';

export default function LiveAgentActivity() {
  const status = useAgentStatus();
  const confidence = Math.round((status.confidence ?? 0.75) * 100);

  return (
    <Card elevation={3} sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" gap={2} mb={1}>
          <Avatar sx={{ bgcolor: 'primary.main' }}><AnalyticsIcon /></Avatar>
          <Typography variant="h6">Live Agent Execution</Typography>
        </Box>

        <Typography variant="body2" paragraph>
          {status.currentTask ?? 'Idle — awaiting instructions'}
        </Typography>

        <LinearProgress variant="determinate" value={confidence} sx={{ mt: 1, mb: 1 }} />
        <Typography variant="caption">Confidence: {confidence}%</Typography>

        <Box mt={2} sx={{ bgcolor: 'grey.100', minHeight: 140, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Typography variant="caption" color="text.secondary">Live device screenshot (placeholder)</Typography>
        </Box>
      </CardContent>
    </Card>
  );
}

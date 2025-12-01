import React from 'react';
import { Paper, Typography, Box } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

const sample = [
  { episode: 0, success_rate: 0.23 },
  { episode: 50, success_rate: 0.67 },
  { episode: 100, success_rate: 0.84 },
  { episode: 150, success_rate: 0.947 },
];

export default function PerformanceCharts({ data = sample }) {
  return (
    <Paper sx={{ p: 2 }} elevation={2}>
      <Typography variant="h6" gutterBottom>Learning Progression</Typography>
      <Box sx={{ overflowX: 'auto' }}>
        <LineChart width={600} height={260} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="episode" />
          <YAxis domain={[0, 1]} />
          <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
          <Line type="monotone" dataKey="success_rate" stroke="#2196F3" strokeWidth={3} dot={false} />
        </LineChart>
      </Box>
    </Paper>
  );
}

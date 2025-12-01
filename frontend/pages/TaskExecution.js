import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Card, 
  CardContent, 
  Grid, 
  Button, 
  TextField, 
  MenuItem, 
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Chip,
  LinearProgress,
  Paper,
  IconButton,
  Collapse,
  useTheme,
  alpha,
} from '@mui/material';
import { 
  PlayArrow, 
  Stop, 
  Refresh,
  CheckCircle,
  RadioButtonChecked,
  Schedule,
  Visibility,
  TouchApp,
  VerifiedUser,
  Psychology,
  ExpandMore,
  ExpandLess,
  Speed,
  Android,
  Apple,
} from '@mui/icons-material';

const TaskExecution = () => {
  const theme = useTheme();
  const [selectedDevice, setSelectedDevice] = useState('');
  const [taskInstruction, setTaskInstruction] = useState('');
  const [maxSteps, setMaxSteps] = useState(10);
  const [activeTask, setActiveTask] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const devices = [
    { id: 'android_pixel_7_1', name: 'Pixel 7 (Android)', platform: 'Android', status: 'active' },
    { id: 'iphone_14_1', name: 'iPhone 14 (iOS)', platform: 'iOS', status: 'active' },
    { id: 'android_galaxy_s23', name: 'Galaxy S23 (Android)', platform: 'Android', status: 'busy' },
    { id: 'iphone_13_1', name: 'iPhone 13 (iOS)', platform: 'iOS', status: 'active' },
  ];

  const sampleTasks = [
    'Send money to John via Venmo',
    'Book an Uber ride to the airport',
    'Order pizza from DoorDash',
    'Check calendar and create meeting',
    'Send WhatsApp message to team',
    'Post a photo on Instagram',
    'Reply to latest email',
    'Add item to shopping cart on Amazon',
  ];

  const taskSteps = [
    { 
      label: 'Task Planning', 
      icon: <Psychology />, 
      description: 'LLM analyzing task and generating action plan',
      status: 'completed',
      duration: '0.8s',
    },
    { 
      label: 'UI Perception', 
      icon: <Visibility />, 
      description: 'Vision model detecting UI elements and screen context',
      status: 'in_progress',
      duration: '1.2s',
    },
    { 
      label: 'Action Execution', 
      icon: <TouchApp />, 
      description: 'Performing tap, swipe, and input actions',
      status: 'pending',
      duration: '--',
    },
    { 
      label: 'Verification', 
      icon: <VerifiedUser />, 
      description: 'Checking task completion and success',
      status: 'pending',
      duration: '--',
    },
    { 
      label: 'Learning Update', 
      icon: <Speed />, 
      description: 'Updating RL policy with task results',
      status: 'pending',
      duration: '--',
    },
  ];

  useEffect(() => {
    if (activeTask && currentStep < taskSteps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prev => prev + 1);
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [activeTask, currentStep, taskSteps.length]);

  const handleStartTask = () => {
    setActiveTask({
      instruction: taskInstruction,
      device: selectedDevice,
      status: 'in_progress',
      startTime: new Date().toLocaleTimeString(),
      progress: 0,
    });
    setCurrentStep(0);
  };

  const handleStopTask = () => {
    setActiveTask(null);
    setCurrentStep(0);
  };

  const getStepStatus = (index) => {
    if (index < currentStep) return 'completed';
    if (index === currentStep) return 'in_progress';
    return 'pending';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return theme.palette.info.main;
      case 'in_progress':
        return theme.palette.primary.main;
      case 'pending':
        return theme.palette.text.disabled;
      default:
        return theme.palette.text.secondary;
    }
  };

  return (
    <Box sx={{ pt: 10, pb: 4, minHeight: '100vh' }}>
      <Container maxWidth="xl">
        {/* Header */}
        <Box sx={{ mb: 4 }}>
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
            Task Execution Center
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Build and execute AI-powered automation tasks
          </Typography>
        </Box>

        <Grid container spacing={3}>
          {/* Left Panel - Task Builder */}
          <Grid item xs={12} lg={5}>
            <Card 
              sx={{ 
                height: '100%',
                background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(theme.palette.secondary.main, 0.05)} 100%)`,
              }}
            >
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h5" fontWeight="600" sx={{ mb: 3, color: theme.palette.primary.main }}>
                  Task Builder
        </Typography>

                {/* Device Selection */}
            <TextField
              select
              label="Select Device"
              value={selectedDevice}
              onChange={(e) => setSelectedDevice(e.target.value)}
              fullWidth
                  sx={{ mb: 3 }}
                  InputProps={{
                    sx: {
                      '& .MuiSelect-select': {
                        display: 'flex',
                        alignItems: 'center',
                      }
                    }
                  }}
                >
                  {devices.map((device) => (
                    <MenuItem key={device.id} value={device.id}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                        {device.platform === 'iOS' ? <Apple /> : <Android />}
                        <Typography>{device.name}</Typography>
                        <Chip 
                          label={device.status} 
                          size="small" 
                          sx={{ 
                            ml: 'auto',
                            bgcolor: alpha(theme.palette.info.main, 0.1),
                            color: theme.palette.info.main,
                          }} 
                        />
                      </Box>
                    </MenuItem>
                  ))}
            </TextField>

                {/* Task Instruction */}
            <TextField
              select
                  label="Task Instruction"
              value={taskInstruction}
              onChange={(e) => setTaskInstruction(e.target.value)}
              fullWidth
                  helperText="Select a sample task or type your own"
              sx={{ mb: 2 }}
            >
              {sampleTasks.map((task) => (
                <MenuItem key={task} value={task}>{task}</MenuItem>
              ))}
            </TextField>

                {/* Custom Task Input */}
                <TextField
                  label="Or Enter Custom Task"
                  value={taskInstruction}
                  onChange={(e) => setTaskInstruction(e.target.value)}
                  fullWidth
                  multiline
                  rows={3}
                  placeholder="Describe the task in natural language..."
                  sx={{ mb: 3 }}
                />

                {/* Advanced Settings Toggle */}
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  endIcon={showAdvanced ? <ExpandLess /> : <ExpandMore />}
                  sx={{ mb: 2 }}
                >
                  Advanced Settings
                </Button>

                <Collapse in={showAdvanced}>
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      Max Steps: {maxSteps}
                    </Typography>
                    <TextField
                      type="number"
                      value={maxSteps}
                      onChange={(e) => setMaxSteps(parseInt(e.target.value))}
                      fullWidth
                      inputProps={{ min: 1, max: 50 }}
                      sx={{ mb: 2 }}
                    />
                  </Box>
                </Collapse>

                {/* Action Buttons */}
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button 
                    variant="contained" 
                    fullWidth
                    size="large"
                    startIcon={<PlayArrow />} 
                    onClick={handleStartTask} 
                    disabled={!selectedDevice || !taskInstruction || activeTask}
                    sx={{
                      py: 1.5,
                      background: 'linear-gradient(135deg, #1e88e5 0%, #1565c0 100%)',
                      '&:hover': {
                        background: 'linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%)',
                      },
                    }}
                  >
              Start Task
            </Button>
                  {activeTask && (
                    <Button 
                      variant="outlined" 
                      color="error"
                      startIcon={<Stop />} 
                      onClick={handleStopTask}
                      sx={{ py: 1.5 }}
                    >
                      Stop
                    </Button>
                  )}
                </Box>

                {/* Task Info */}
                {activeTask && (
                  <Paper sx={{ mt: 3, p: 2, bgcolor: alpha(theme.palette.info.main, 0.05) }}>
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                      <strong>Started:</strong> {activeTask.startTime}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Device:</strong> {devices.find(d => d.id === activeTask.device)?.name}
                    </Typography>
                  </Paper>
                )}
          </CardContent>
        </Card>
          </Grid>

          {/* Right Panel - Execution Timeline */}
          <Grid item xs={12} lg={7}>
            <Card sx={{ height: '100%' }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h5" fontWeight="600" sx={{ mb: 3, color: theme.palette.primary.main }}>
                  {activeTask ? 'Live Execution Timeline' : 'Execution Preview'}
                </Typography>

                {!activeTask ? (
                  <Box sx={{ textAlign: 'center', py: 8 }}>
                    <Schedule sx={{ fontSize: 80, color: theme.palette.text.disabled, mb: 2 }} />
                    <Typography variant="h6" color="text.secondary">
                      Configure and start a task to see live execution
                    </Typography>
                  </Box>
                ) : (
                  <Box>
                    {/* Overall Progress */}
                    <Paper sx={{ p: 2, mb: 3, bgcolor: alpha(theme.palette.primary.main, 0.05) }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" fontWeight="600">
                          Overall Progress
                        </Typography>
                        <Typography variant="body2" fontWeight="600" color="primary">
                          {Math.min(((currentStep + 1) / taskSteps.length) * 100, 100).toFixed(0)}%
                        </Typography>
                      </Box>
                      <LinearProgress 
                        variant="determinate" 
                        value={((currentStep + 1) / taskSteps.length) * 100} 
                        sx={{ 
                          height: 10, 
                          borderRadius: 5,
                          bgcolor: alpha(theme.palette.primary.main, 0.1),
                          '& .MuiLinearProgress-bar': {
                            background: 'linear-gradient(90deg, #1e88e5 0%, #42a5f5 100%)',
                          }
                        }} 
                      />
                    </Paper>

                    {/* Task Instruction Display */}
                    <Paper sx={{ p: 2, mb: 3, bgcolor: alpha(theme.palette.info.main, 0.05), border: `2px solid ${alpha(theme.palette.info.main, 0.2)}` }}>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                        TASK INSTRUCTION
                      </Typography>
                      <Typography variant="body1" fontWeight="600">
                        {activeTask.instruction}
                      </Typography>
                    </Paper>

                    {/* Stepper Timeline */}
                    <Stepper activeStep={currentStep} orientation="vertical">
                      {taskSteps.map((step, index) => {
                        const status = getStepStatus(index);
                        return (
                          <Step key={step.label} completed={status === 'completed'}>
                            <StepLabel
                              StepIconComponent={() => (
                                <Box
                                  sx={{
                                    width: 40,
                                    height: 40,
                                    borderRadius: '50%',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    bgcolor: alpha(getStatusColor(status), 0.15),
                                    color: getStatusColor(status),
                                    border: `2px solid ${getStatusColor(status)}`,
                                    animation: status === 'in_progress' ? 'pulse-blue 2s infinite' : 'none',
                                  }}
                                >
                                  {status === 'completed' ? <CheckCircle /> : step.icon}
                                </Box>
                              )}
                            >
                              <Box>
                                <Typography variant="body1" fontWeight="600">
                                  {step.label}
                                </Typography>
                                <Chip 
                                  label={status.replace('_', ' ').toUpperCase()} 
                                  size="small" 
                                  sx={{ 
                                    mt: 0.5,
                                    bgcolor: alpha(getStatusColor(status), 0.15),
                                    color: getStatusColor(status),
                                    fontWeight: 600,
                                  }} 
                                />
                              </Box>
                            </StepLabel>
                            <StepContent>
                              <Paper sx={{ p: 2, bgcolor: alpha(theme.palette.background.paper, 0.5) }}>
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                  {step.description}
                                </Typography>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <Schedule sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                                  <Typography variant="caption" color="text.secondary">
                                    Duration: {step.duration}
                                  </Typography>
                                </Box>
                              </Paper>
                            </StepContent>
                          </Step>
                        );
                      })}
                    </Stepper>

                    {/* Completion Message */}
                    {currentStep >= taskSteps.length - 1 && (
                      <Paper 
                        sx={{ 
                          mt: 3, 
                          p: 3, 
                          textAlign: 'center',
                          bgcolor: alpha(theme.palette.info.main, 0.1),
                          border: `2px solid ${theme.palette.info.main}`,
                        }}
                      >
                        <CheckCircle sx={{ fontSize: 60, color: theme.palette.info.main, mb: 2 }} />
                        <Typography variant="h6" fontWeight="600" color="info.main" sx={{ mb: 1 }}>
                          Task Completed Successfully!
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          All steps executed without errors
                        </Typography>
                      </Paper>
                    )}
                  </Box>
                )}
            </CardContent>
          </Card>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default TaskExecution;

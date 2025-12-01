import React, { useState, useEffect } from 'react';
import {
  Fab,
  Dialog,
  DialogContent,
  Typography,
  Box,
  IconButton,
  Paper,
  CircularProgress,
  Alert
} from '@mui/material';
import { Mic, Close } from '@mui/icons-material';

const VoiceControl = ({ onVoiceCommand }) => {
  const [isListening, setIsListening] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [recognition, setRecognition] = useState(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognitionInstance = new SpeechRecognition();
      
      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = true;
      recognitionInstance.lang = 'en-US';

      recognitionInstance.onstart = () => {
        setIsListening(true);
      };

      recognitionInstance.onresult = (event) => {
        const current = event.resultIndex;
        const t = event.results[current][0].transcript;
        setTranscript(t);

        if (event.results[current].isFinal) {
          onVoiceCommand && onVoiceCommand(t);
          setDialogOpen(false);
          setTranscript('');
        }
      };

      recognitionInstance.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsListening(false);
        setDialogOpen(false);
      };

      recognitionInstance.onend = () => {
        setIsListening(false);
      };

      setRecognition(recognitionInstance);
    }
  }, [onVoiceCommand]);

  const startListening = () => {
    if (recognition) {
      setDialogOpen(true);
      recognition.start();
    }
  };

  const stopListening = () => {
    if (recognition) {
      recognition.stop();
    }
    setDialogOpen(false);
    setTranscript('');
  };

  return (
    <>
      <Fab
        color="primary"
        aria-label="voice control"
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
          zIndex: 1000
        }}
        onClick={startListening}
      >
        <Mic />
      </Fab>

      <Dialog
        open={dialogOpen}
        onClose={stopListening}
        maxWidth="sm"
        fullWidth
        sx={{
          '& .MuiDialog-paper': {
            borderRadius: 3,
            p: 2
          }
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6">Voice Command</Typography>
          <IconButton onClick={stopListening}>
            <Close />
          </IconButton>
        </Box>

        <DialogContent sx={{ textAlign: 'center' }}>
          <Box sx={{ mb: 3 }}>
            <CircularProgress 
              size={80} 
              sx={{ 
                color: isListening ? 'primary.main' : 'grey.400',
                mb: 2
              }}
            />
            <Typography variant="h6" gutterBottom>
              {isListening ? 'Listening...' : 'Click to speak'}
            </Typography>
          </Box>

          {transcript && (
            <Paper elevation={1} sx={{ p: 2, mb: 2, bgcolor: 'grey.50' }}>
              <Typography variant="body1">
                "{transcript}"
              </Typography>
            </Paper>
          )}

          <Alert severity="info">
            Try saying: "Start a new task", "Show device status", or "Pause current task"
          </Alert>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default VoiceControl;

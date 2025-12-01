import { useState, useEffect } from 'react';

export default function useAgentStatus(wsUrl = 'ws://localhost:5000/ws/agent-status') {
  const [status, setStatus] = useState({
    isActive: false,
    currentTask: null,
    progress: 0,
    confidence: 0.75,
    lastAction: null,
  });

  useEffect(() => {
    let ws;
    let fallbackInterval;
    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      ws = null;
    }

    if (ws) {
      ws.onopen = () => setStatus(prev => ({ ...prev, isActive: true }));
      ws.onmessage = (evt) => {
        try {
          const payload = JSON.parse(evt.data);
          setStatus(prev => ({ ...prev, ...payload }));
        } catch (err) {
          // ignore malformed payload
        }
      };
      ws.onclose = () => setStatus(prev => ({ ...prev, isActive: false }));
    } else {
      // fallback: simulate updates so demo UI looks alive
      fallbackInterval = setInterval(() => {
        setStatus(prev => ({
          ...prev,
          progress: Math.min(100, (prev.progress || 0) + Math.floor(Math.random() * 8)),
          confidence: Math.max(0.4, Math.min(0.99, (prev.confidence || 0.75) + (Math.random() - 0.5) * 0.05)),
          currentTask: prev.currentTask || 'Analyzing screen...'
        }));
      }, 1500);
    }

    return () => {
      if (ws) ws.close();
      if (fallbackInterval) clearInterval(fallbackInterval);
    };
  }, [wsUrl]);

  return status;
}

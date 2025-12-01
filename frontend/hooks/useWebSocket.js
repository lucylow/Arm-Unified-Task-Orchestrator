/**
 * WebSocket Hook for Real-time Updates
 * Connects to AutoRL backend WebSocket for live task execution updates
 */

import { useEffect, useRef, useState, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:5000/ws';
const RECONNECT_DELAY = 3000; // 3 seconds

export function useWebSocket(onMessage) {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState(null);
  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const shouldReconnectRef = useRef(true);

  const connect = useCallback(() => {
    try {
      console.log('🔌 Connecting to WebSocket:', WS_URL);
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        // Clear any pending reconnection attempts
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
          reconnectTimeoutRef.current = null;
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('📨 WebSocket message:', data);
          setLastMessage(data);
          if (onMessage) {
            onMessage(data);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
      };

      ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        setIsConnected(false);
        wsRef.current = null;

        // Attempt to reconnect if should reconnect is true
        if (shouldReconnectRef.current) {
          console.log(`⏳ Reconnecting in ${RECONNECT_DELAY/1000}s...`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, RECONNECT_DELAY);
        }
      };

      wsRef.current = ws;
    } catch (error) {
      console.error('Error creating WebSocket connection:', error);
    }
  }, [onMessage]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      return true;
    } else {
      console.warn('WebSocket is not connected. Cannot send message.');
      return false;
    }
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    disconnect,
    reconnect: connect
  };
}

export default useWebSocket;


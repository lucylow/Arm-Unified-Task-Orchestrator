import React, { useRef, useState, useEffect } from 'react';
import DeviceSimulator from './DeviceSimulator';
import MobileScreen from './MobileScreen';
import MobileController from './MobileController';
import CrossAppDemo from './CrossAppDemo';
import './mobile.css';

/*
Example integration for Dashboard.jsx
Replace your iframe block with this component
*/

export default function MobileDemoPanel({ mode = 'single' }) {
  const mobileRef = useRef(null);
  const [events, setEvents] = useState([]);
  const [attentionData, setAttentionData] = useState({});
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  const onEvent = (ev) => {
    setEvents(s => [ev, ...s].slice(0, 120));
    console.log('[AutoRL Event]', ev);
  };

  // WebSocket connection to backend
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        const ws = new WebSocket('ws://localhost:8000/ws/metrics?client_id=dashboard');
        
        ws.onopen = () => {
          console.log('✅ WebSocket connected to AutoRL backend');
          setWsConnected(true);
          onEvent({ 
            event: 'websocket_connected', 
            text: 'Connected to AutoRL backend', 
            timestamp: Date.now() 
          });
        };
        
        ws.onmessage = (message) => {
          try {
            const data = JSON.parse(message.data);
            console.log('[WebSocket Message]', data);
            
            // Handle different message types
            switch (data.type) {
              case 'connection_established':
                onEvent({ 
                  event: 'system', 
                  text: 'Backend connection established', 
                  timestamp: data.timestamp 
                });
                break;
                
              case 'task_update':
                onEvent({
                  event: 'task_update',
                  text: `Task ${data.task_id}: ${data.status} (${(data.progress * 100).toFixed(0)}%)`,
                  timestamp: Date.now()
                });
                break;
                
              case 'heartbeat':
                // Silent heartbeat
                break;
                
              case 'attention':
                // Update attention heatmap
                if (data.attention_data) {
                  setAttentionData(data.attention_data);
                }
                break;
                
              case 'gesture':
                // Trigger visual gesture
                window.dispatchEvent(new CustomEvent('autorlgesture', { 
                  detail: data 
                }));
                break;
                
              default:
                onEvent({
                  event: data.type || 'message',
                  text: data.text || JSON.stringify(data),
                  timestamp: data.timestamp || Date.now()
                });
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };
        
        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setWsConnected(false);
          onEvent({ 
            event: 'websocket_error', 
            text: 'WebSocket connection error', 
            timestamp: Date.now() 
          });
        };
        
        ws.onclose = () => {
          console.log('WebSocket disconnected, attempting reconnect...');
          setWsConnected(false);
          // Attempt reconnection after 5 seconds
          setTimeout(connectWebSocket, 5000);
        };
        
        wsRef.current = ws;
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setTimeout(connectWebSocket, 5000);
      }
    };
    
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Send ping to backend periodically
  useEffect(() => {
    if (!wsConnected || !wsRef.current) return;
    
    const interval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }
    }, 30000);
    
    return () => clearInterval(interval);
  }, [wsConnected]);

  if (mode === 'cross-app') {
    return (
      <div style={{ padding: 20 }}>
        <CrossAppDemo onEvent={onEvent} />
        <EventLog events={events} wsConnected={wsConnected} />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', gap: 20, padding: 20 }}>
      <div style={{ flex: '0 0 360px' }}>
        <DeviceSimulator>
          <MobileScreen 
            ref={mobileRef} 
            initialApp="banking_v1" 
            onAction={(action) => {
              console.log('[Mobile Action]', action);
              onEvent({ 
                event: 'mobile_action', 
                text: `${action.type}: ${action.elementId || 'screen'}`, 
                timestamp: Date.now() 
              });
            }}
            attentionData={attentionData}
          />
        </DeviceSimulator>

        <div style={{ marginTop: 12, display: 'flex', gap: 8, flexDirection: 'column' }}>
          <MobileController mobileRef={mobileRef} onEvent={onEvent} />
          
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button 
              className="btn" 
              style={{ background: '#2b6bff' }}
              onClick={() => {
                mobileRef.current?.switchApp('banking_v2');
                onEvent({
                  event: 'app_update',
                  text: 'Switched to banking_v2 (simulated app update)',
                  timestamp: Date.now()
                });
              }}
            >
              🔄 Force App Update
            </button>
            
            <button 
              className="btn" 
              style={{ background: '#00cc66' }}
              onClick={() => {
                mobileRef.current?.switchApp('calendar');
                onEvent({
                  event: 'app_switch',
                  text: 'Switched to Calendar app',
                  timestamp: Date.now()
                });
              }}
            >
              📅 Calendar
            </button>
            
            <button 
              className="btn" 
              style={{ background: '#9966ff' }}
              onClick={() => {
                mobileRef.current?.switchApp('chat');
                onEvent({
                  event: 'app_switch',
                  text: 'Switched to Chat app',
                  timestamp: Date.now()
                });
              }}
            >
              💬 Chat
            </button>
          </div>
        </div>
      </div>

      <div style={{ flex: 1 }}>
        <EventLog events={events} wsConnected={wsConnected} />
      </div>
    </div>
  );
}

// Event Log Component
function EventLog({ events, wsConnected }) {
  const getEventStyle = (eventType) => {
    const styles = {
      perception: { icon: '👁️', color: '#00e3ff' },
      planning: { icon: '🧠', color: '#7a6bff' },
      action: { icon: '⚡', color: '#00ff99' },
      completed: { icon: '✅', color: '#00ff99' },
      error: { icon: '❌', color: '#ff4444' },
      warning: { icon: '⚠️', color: '#ff9944' },
      recovery_analyze: { icon: '🔍', color: '#00ccff' },
      recovery_plan: { icon: '📋', color: '#00ccff' },
      recovery_execute: { icon: '🔧', color: '#00ccff' },
      recovered: { icon: '✨', color: '#00ff99' },
      memory_saved: { icon: '💾', color: '#9966ff' },
      cross_app_start: { icon: '🔀', color: '#00e3ff' },
      app_update: { icon: '🔄', color: '#ff9944' },
      app_switch: { icon: '📱', color: '#00ccff' },
      websocket_connected: { icon: '🔌', color: '#00ff99' },
      websocket_error: { icon: '🔌', color: '#ff4444' },
      default: { icon: '•', color: '#9fcfe6' }
    };
    
    return styles[eventType] || styles.default;
  };

  return (
    <div style={{
      background: 'rgba(4,20,39,0.6)',
      border: '1px solid rgba(0,229,255,0.2)',
      borderRadius: 12,
      padding: 16,
      height: '750px',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
        paddingBottom: 12,
        borderBottom: '1px solid rgba(0,229,255,0.1)'
      }}>
        <h3 style={{ color: '#00e5ff', margin: 0, fontSize: 16 }}>
          Agent Activity Log
        </h3>
        <div style={{ 
          fontSize: 12, 
          color: wsConnected ? '#00ff99' : '#ff4444',
          display: 'flex',
          alignItems: 'center',
          gap: 6
        }}>
          <div style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: wsConnected ? '#00ff99' : '#ff4444',
            animation: wsConnected ? 'pulse 2s infinite' : 'none'
          }} />
          {wsConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      <div style={{
        flex: 1,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 6
      }}>
        {events.length === 0 && (
          <div style={{ 
            color: '#9fcfe6', 
            fontSize: 13, 
            textAlign: 'center',
            marginTop: 20 
          }}>
            No events yet. Start a mobile demo to see agent activity.
          </div>
        )}
        
        {events.map((event, idx) => {
          const style = getEventStyle(event.event);
          const time = new Date(event.timestamp).toLocaleTimeString();
          
          return (
            <div
              key={idx}
              style={{
                background: 'rgba(6,25,43,0.5)',
                border: '1px solid rgba(0,229,255,0.05)',
                borderRadius: 8,
                padding: '8px 12px',
                fontSize: 13,
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10,
                animation: idx === 0 ? 'slideIn 0.3s ease-out' : 'none'
              }}
            >
              <span style={{ fontSize: 16 }}>{style.icon}</span>
              <div style={{ flex: 1 }}>
                <div style={{ color: style.color, fontWeight: 600, marginBottom: 2 }}>
                  {event.event.replace(/_/g, ' ').toUpperCase()}
                </div>
                <div style={{ color: '#cceeff' }}>{event.text}</div>
                {event.plan && (
                  <div style={{ 
                    marginTop: 4, 
                    fontSize: 12, 
                    color: '#9fcfe6',
                    fontFamily: 'monospace' 
                  }}>
                    {event.plan.join(' → ')}
                  </div>
                )}
              </div>
              <div style={{ fontSize: 11, color: '#6b8fa1' }}>{time}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// Add animation keyframes via CSS
const style = document.createElement('style');
style.textContent = `
  @keyframes slideIn {
    from {
      opacity: 0;
      transform: translateX(-10px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
`;
document.head.appendChild(style);


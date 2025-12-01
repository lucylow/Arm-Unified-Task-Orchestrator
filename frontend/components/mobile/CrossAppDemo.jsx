import React, { useState } from 'react';
import './mobile.css';

/*
CrossAppDemo - Demonstrates AI agent coordinating across multiple apps
Shows: Calendar → Chat → Email workflow with visual data flow
*/

export default function CrossAppDemo({ onEvent }) {
  const [activeApp, setActiveApp] = useState('calendar');
  const [dataFlow, setDataFlow] = useState({ from: null, to: null, active: false });
  const [brainActivity, setBrainActivity] = useState(0);
  const [showError, setShowError] = useState(false);
  const [showRecovery, setShowRecovery] = useState(false);

  const pulseAnimation = {
    scale: [1, 1.05, 1],
    opacity: [0.6, 1, 0.6],
  };

  const triggerDataFlow = (from, to) => {
    setDataFlow({ from, to, active: true });
    setBrainActivity(prev => prev + 1);
    
    setTimeout(() => {
      setDataFlow({ from: null, to: null, active: false });
    }, 2000);
  };

  const triggerError = () => {
    setShowError(true);
    onEvent?.({ event: 'error', text: 'Connection timeout detected', timestamp: Date.now() });
    
    setTimeout(() => {
      setShowError(false);
      setShowRecovery(true);
      onEvent?.({ event: 'recovery', text: 'Analyzing alternate routes...', timestamp: Date.now() });
      
      setTimeout(() => {
        setShowRecovery(false);
        onEvent?.({ event: 'recovered', text: 'Recovered successfully!', timestamp: Date.now() });
      }, 2000);
    }, 1500);
  };

  return (
    <div className="cross-app-container">
      {/* AI Brain Core */}
      <div className="ai-brain">
        <motion.div 
          className="ai-brain-core"
          animate={pulseAnimation}
          transition={{ duration: 3, repeat: Infinity }}
        >
          <svg width="80" height="80" viewBox="0 0 80 80">
            <circle 
              cx="40" 
              cy="40" 
              r="35" 
              stroke="#00e5ff" 
              strokeWidth="2" 
              fill="none"
              strokeDasharray="4 4"
            />
            <path
              d="M40,10 Q60,30 40,70 Q20,30 40,10Z"
              stroke="#00b3ff"
              strokeWidth="2"
              fill="none"
            />
          </svg>
          <div style={{ 
            position: 'absolute', 
            fontSize: 12, 
            color: '#00e5ff',
            fontWeight: 600
          }}>
            AutoRL
          </div>
        </motion.div>
        <div className="ai-brain-label">AI Agent Core</div>
      </div>

      {/* Apps Row */}
      <div style={{ display: 'flex', gap: 20, marginTop: 20 }}>
        {/* Calendar App */}
        <motion.div 
          className="cross-app-phone"
          animate={{ 
            borderColor: activeApp === 'calendar' ? '#00e5ff' : 'rgba(0,229,255,0.2)',
            boxShadow: activeApp === 'calendar' 
              ? '0 10px 40px rgba(0,229,255,0.3)' 
              : '0 10px 30px rgba(0,229,255,0.1)'
          }}
          transition={{ duration: 0.3 }}
        >
          <div className="cross-app-title">📅 Calendar</div>
          
          <div className="cross-app-event-card">
            <div className="cross-app-event-title">Team Meeting</div>
            <div className="cross-app-event-time">10:00 AM - Zoom</div>
          </div>
          
          <button 
            className="mobile-primary" 
            onClick={() => {
              setActiveApp('chat');
              triggerDataFlow('calendar', 'chat');
              onEvent?.({ 
                event: 'action', 
                text: 'Sending invite from Calendar to Chat', 
                timestamp: Date.now() 
              });
            }}
            style={{ marginTop: 'auto' }}
          >
            Send Invite
          </button>
        </motion.div>

        {/* Chat App */}
        <motion.div 
          className="cross-app-phone"
          animate={{ 
            borderColor: activeApp === 'chat' ? '#00e5ff' : 'rgba(0,229,255,0.2)',
            boxShadow: activeApp === 'chat' 
              ? '0 10px 40px rgba(0,229,255,0.3)' 
              : '0 10px 30px rgba(0,229,255,0.1)'
          }}
          transition={{ duration: 0.3 }}
        >
          <div className="cross-app-title">💬 Messages</div>
          
          <div className="cross-app-chat-bubble">
            <div className="cross-app-chat-text">Invitation Sent ✅</div>
          </div>
          
          <button 
            className="mobile-secondary" 
            onClick={() => {
              setActiveApp('email');
              triggerDataFlow('chat', 'email');
              onEvent?.({ 
                event: 'action', 
                text: 'Forwarding confirmation to Email', 
                timestamp: Date.now() 
              });
            }}
            style={{ marginTop: 'auto' }}
          >
            Forward to Email
          </button>
        </motion.div>

        {/* Email App */}
        <motion.div 
          className="cross-app-phone"
          animate={{ 
            borderColor: activeApp === 'email' ? '#00e5ff' : 'rgba(0,229,255,0.2)',
            boxShadow: activeApp === 'email' 
              ? '0 10px 40px rgba(0,229,255,0.3)' 
              : '0 10px 30px rgba(0,229,255,0.1)'
          }}
          transition={{ duration: 0.3 }}
        >
          <div className="cross-app-title">📧 Email</div>
          
          <div className="cross-app-event-card">
            <div className="cross-app-event-title">Meeting Confirmation</div>
            <div style={{ color: '#9fcfe6', fontSize: 13, marginTop: 5 }}>
              To: team@company.com
            </div>
          </div>
          
          <button 
            className="mobile-confirm" 
            onClick={() => {
              onEvent?.({ 
                event: 'completed', 
                text: 'Cross-app workflow completed successfully!', 
                timestamp: Date.now(),
                success: true 
              });
            }}
            style={{ marginTop: 'auto' }}
          >
            Send Email
          </button>
        </motion.div>
      </div>

      {/* Data Flow Beams */}
      {dataFlow.active && (
        <motion.div
          className="data-flow-beam"
          initial={{ width: 0, opacity: 0 }}
          animate={{ width: 250, opacity: 1 }}
          exit={{ opacity: 0 }}
          style={{
            position: 'absolute',
            top: '50%',
            left: dataFlow.from === 'calendar' ? '30%' : '50%',
            transformOrigin: 'left center'
          }}
        />
      )}

      {/* Error Popup */}
      {showError && (
        <motion.div
          className="error-popup"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.8 }}
          style={{
            position: 'absolute',
            top: '40%',
            left: '50%',
            transform: 'translate(-50%, -50%)'
          }}
        >
          <div className="error-title">⚠️ Workflow Error</div>
          <div className="error-message">Connection timeout — Re-planning route...</div>
        </motion.div>
      )}

      {/* Recovery Flow */}
      {showRecovery && (
        <motion.div
          className="recovery-flow"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
        >
          <div className="recovery-text">
            🧠 Pause → Analyze → Plan Alternate → Resume
          </div>
        </motion.div>
      )}

      {/* Control Buttons */}
      <div style={{ 
        marginTop: 30, 
        display: 'flex', 
        gap: 10, 
        justifyContent: 'center' 
      }}>
        <button className="btn" onClick={() => {
          setActiveApp('calendar');
          triggerDataFlow('calendar', 'chat');
        }}>
          ▶️ Start Workflow
        </button>
        
        <button className="btn" style={{ background: '#cc6600' }} onClick={triggerError}>
          ⚠️ Inject Error
        </button>
      </div>

      {/* HUD */}
      <div className="mobile-hud">
        <div className="hud-title">Cross-App Agent Status</div>
        <div className="hud-text">Active App: {activeApp}</div>
        <div className="hud-text">Brain Activity: {brainActivity} operations</div>
        <div className="hud-text">Status: {showError ? '❌ Error' : showRecovery ? '🔄 Recovering' : '✅ Operational'}</div>
      </div>
    </div>
  );
}


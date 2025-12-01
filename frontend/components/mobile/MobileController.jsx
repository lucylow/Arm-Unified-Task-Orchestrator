import React, { useState } from 'react';

/*
Props:
- mobileRef: ref to MobileScreen
- onEvent: callback(event) to bubble events to dashboard log
- simulateRun(options) // triggers perception->plan->execute flow visually
*/

export default function MobileController({ mobileRef, onEvent, disabled = false }) {
  const [isRunning, setIsRunning] = useState(false);
  
  const runSim = async ({ injectUpdate = false, injectError = false } = {}) => {
    if (isRunning) return;
    setIsRunning(true);
    
    onEvent?.({ event: 'perception', text: 'Capturing screenshot and OCR...', timestamp: Date.now() });

    // small delay
    await new Promise(r => setTimeout(r, 700));

    onEvent?.({ 
      event: 'planning', 
      text: 'LLM planning: build action sequence', 
      plan: ['tap_send', 'type_amount', 'select_recipient', 'confirm'], 
      timestamp: Date.now() 
    });

    await new Promise(r => setTimeout(r, 900));

    // find target element in current app
    const elBox = mobileRef.current?.getElementBox ? mobileRef.current.getElementBox('send-money') : null;
    
    // if the element isn't found (e.g., after UI update), fallback to semantic search & planning
    if (!elBox || injectUpdate) {
      onEvent?.({ 
        event: 'warning', 
        text: 'Primary button not found. Searching by semantics and iconography...',
        timestamp: Date.now()
      });
      
      await new Promise(r => setTimeout(r, 700));
      
      // simulate searching and tapping new button (data-role="payment")
      const newBox = mobileRef.current?.getElementBox && mobileRef.current.getElementBox('pay');
      
      if (newBox) {
        // animate tap
        window.dispatchEvent(new CustomEvent('autorlgesture', { 
          detail: { type: 'tap', x: newBox.x + newBox.w/2, y: newBox.y + newBox.h/2 } 
        }));
        onEvent?.({ 
          event: 'action', 
          text: 'Tap pay (discovered by semantic match)', 
          timestamp: Date.now() 
        });
      } else if (injectError) {
        // show error and recovery
        onEvent?.({ 
          event: 'error', 
          text: 'UI element not found — entering recovery', 
          timestamp: Date.now() 
        });
        await doRecovery();
        setIsRunning(false);
        return;
      }
    } else {
      // animate a tap at the element
      window.dispatchEvent(new CustomEvent('autorlgesture', { 
        detail: { type: 'tap', x: elBox.x + elBox.w/2, y: elBox.y + elBox.h/2 } 
      }));
      onEvent?.({ event: 'action', text: 'Tap Send Money', timestamp: Date.now() });
    }

    // show typing
    if (mobileRef.current) {
      await new Promise(r => setTimeout(r, 600));
      
      // type amount
      await mobileRef.current.type('amount', '$20.00');
      window.dispatchEvent(new CustomEvent('autorlgesture', { 
        detail: { type: 'type', x: 180, y: 320, text: '$20.00' } 
      }));
      onEvent?.({ event: 'action', text: 'Type amount $20.00', timestamp: Date.now() });
      
      // simulate pressing confirm
      const box = mobileRef.current.getElementBox && mobileRef.current.getElementBox('confirm');
      if (box) {
        await new Promise(r => setTimeout(r, 400));
        window.dispatchEvent(new CustomEvent('autorlgesture', { 
          detail: { type: 'tap', x: box.x + box.w/2, y: box.y + box.h/2 } 
        }));
        onEvent?.({ event: 'action', text: 'Tap Confirm', timestamp: Date.now() });
      }
    }

    await new Promise(r => setTimeout(r, 900));
    onEvent?.({ 
      event: 'completed', 
      text: 'Transaction Complete! $20 sent to Jane', 
      timestamp: Date.now(), 
      success: true 
    });
    
    // save memory - visual indicator
    onEvent?.({ event: 'memory_saved', text: 'Episode saved to memory', timestamp: Date.now() });
    
    setIsRunning(false);
  };

  const doRecovery = async () => {
    onEvent?.({ 
      event: 'recovery_analyze', 
      text: 'Analyzing screen for alternate flows', 
      timestamp: Date.now() 
    });
    
    await new Promise(r => setTimeout(r, 800));
    
    onEvent?.({ 
      event: 'recovery_plan', 
      text: 'Plan: open Payments tab → locate Pay button → submit', 
      timestamp: Date.now() 
    });
    
    await new Promise(r => setTimeout(r, 700));
    
    onEvent?.({ 
      event: 'recovery_execute', 
      text: 'Executing recovery plan', 
      timestamp: Date.now() 
    });
    
    // simulate a successful tap at a fallback region center
    window.dispatchEvent(new CustomEvent('autorlgesture', { 
      detail: { type: 'tap', x: 200, y: 380 } 
    }));
    
    await new Promise(r => setTimeout(r, 600));
    
    onEvent?.({ 
      event: 'recovered', 
      text: 'Recovered and resumed', 
      timestamp: Date.now() 
    });
  };

  const runCrossAppDemo = async () => {
    if (isRunning) return;
    setIsRunning(true);
    
    onEvent?.({ 
      event: 'cross_app_start', 
      text: 'Starting cross-app workflow: Calendar → Chat', 
      timestamp: Date.now() 
    });
    
    // Switch to calendar
    if (mobileRef.current?.switchApp) {
      mobileRef.current.switchApp('calendar');
    }
    
    await new Promise(r => setTimeout(r, 1000));
    
    onEvent?.({ 
      event: 'action', 
      text: 'Analyzing calendar event...', 
      timestamp: Date.now() 
    });
    
    await new Promise(r => setTimeout(r, 800));
    
    // Tap send invite
    const inviteBox = mobileRef.current?.getElementBox('send-invite');
    if (inviteBox) {
      window.dispatchEvent(new CustomEvent('autorlgesture', { 
        detail: { type: 'tap', x: inviteBox.x + inviteBox.w/2, y: inviteBox.y + inviteBox.h/2 } 
      }));
      onEvent?.({ event: 'action', text: 'Tap Send Invite', timestamp: Date.now() });
    }
    
    await new Promise(r => setTimeout(r, 1200));
    
    // Switch to chat
    onEvent?.({ 
      event: 'action', 
      text: 'Switching to Messages app...', 
      timestamp: Date.now() 
    });
    
    if (mobileRef.current?.switchApp) {
      mobileRef.current.switchApp('chat');
    }
    
    await new Promise(r => setTimeout(r, 1000));
    
    onEvent?.({ 
      event: 'completed', 
      text: 'Cross-app workflow completed! Invitation sent via chat.', 
      timestamp: Date.now(),
      success: true 
    });
    
    onEvent?.({ event: 'memory_saved', text: 'Workflow episode saved', timestamp: Date.now() });
    
    setIsRunning(false);
  };

  return (
    <div style={{ display:'flex', gap:8, flexWrap: 'wrap' }}>
      <button 
        className="btn" 
        onClick={() => runSim({})}
        disabled={disabled || isRunning}
        style={{ opacity: (disabled || isRunning) ? 0.5 : 1 }}
      >
        {isRunning ? '🔄 Running...' : '▶️ Run Mobile Demo'}
      </button>
      
      <button 
        className="btn" 
        style={{ background:'#0077cc' }} 
        onClick={() => runSim({ injectUpdate: true })}
        disabled={disabled || isRunning}
      >
        🔄 Run + Inject App Update
      </button>
      
      <button 
        className="btn" 
        style={{ background:'#cc6600' }} 
        onClick={() => runSim({ injectError: true })}
        disabled={disabled || isRunning}
      >
        ⚠️ Run + Inject Error
      </button>
      
      <button 
        className="btn" 
        style={{ background:'#00cc66' }} 
        onClick={runCrossAppDemo}
        disabled={disabled || isRunning}
      >
        🔀 Run Cross-App Demo
      </button>
    </div>
  );
}


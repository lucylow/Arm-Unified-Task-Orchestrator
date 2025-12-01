import React, { useRef, useImperativeHandle, forwardRef, useState } from 'react';
import GestureOverlay from './GestureOverlay';
import AttentionHeatmap from './AttentionHeatmap';
import OCROverlay from './OCROverlay';

/*
API (ref):
mobileRef.current.tap(x,y)
mobileRef.current.type(selector, text)
mobileRef.current.swipe(from, to)
mobileRef.current.getElementBox(selector) -> {x,y,w,h}
mobileRef.current.switchApp('banking_v2')
*/

const APPS = {
  banking_v1: {
    id: 'banking_v1',
    title: 'Checking Account',
    balance: '$3,214.00',
    primaryButton: { id: 'send-money', label: 'Send Money' },
    fields: [
      { id: 'recipient', placeholder: 'Recipient: Jane', value: 'Jane' },
      { id: 'amount', placeholder: 'Amount: $20.00', value: '$20.00' }
    ]
  },
  banking_v2: {
    id: 'banking_v2',
    title: 'Payments',
    balance: '$3,214.00',
    primaryButton: { id: 'pay', label: 'Pay' },
    fields: [
      { id: 'payee', placeholder: 'Who are you paying?', value: '' },
      { id: 'amount', placeholder: '$0.00', value: '' }
    ]
  },
  calendar: {
    id: 'calendar',
    title: 'Calendar',
    events: [
      { id: 'meeting', title: 'Team Meeting', time: '10:00 AM', location: 'Zoom' }
    ],
    primaryButton: { id: 'send-invite', label: 'Send Invite' }
  },
  chat: {
    id: 'chat',
    title: 'Messages',
    messages: [
      { id: 1, text: 'Invitation Sent!', from: 'system', timestamp: Date.now() },
      { id: 2, text: 'Meeting confirmed for 10 AM', from: 'user', timestamp: Date.now() + 1000 }
    ]
  },
  email: {
    id: 'email',
    title: 'Email',
    primaryButton: { id: 'compose', label: 'Compose' },
    fields: [
      { id: 'to', placeholder: 'To: team@company.com', value: 'team@company.com' },
      { id: 'subject', placeholder: 'Subject: Meeting Update', value: 'Meeting Update' }
    ]
  }
};

const MobileScreen = forwardRef(({ initialApp = 'banking_v1', onAction, attentionData = {} }, ref) => {
  const [app, setApp] = useState(APPS[initialApp]);
  const rootRef = useRef(null);
  const elementsRef = useRef({}); // map id -> DOM rect or element

  function register(id, el) {
    if (!el) return;
    elementsRef.current[id] = el;
  }

  useImperativeHandle(ref, () => ({
    tap: async (x, y) => {
      // if within phone screen translate coordinates; but for demo we assume callers pass element ids
      if (onAction) onAction({ type: 'tap', x, y });
      return { ok: true };
    },
    type: async (elementId, text) => {
      const el = elementsRef.current[elementId];
      if (el) {
        // set value and dispatch input event to simulate typing
        el.value = text;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        if (onAction) onAction({ type: 'type', elementId, text });
        return { ok: true };
      }
      return { ok: false, error: 'element not found' };
    },
    swipe: async (from, to) => {
      if (onAction) onAction({ type: 'swipe', from, to });
      // visual only
      return { ok: true };
    },
    getElementBox: (id) => {
      const el = elementsRef.current[id];
      if (!el) return null;
      const r = el.getBoundingClientRect();
      const rootR = rootRef.current.getBoundingClientRect();
      return {
        x: r.left - rootR.left,
        y: r.top - rootR.top,
        w: r.width,
        h: r.height
      };
    },
    switchApp: (appId) => {
      setApp(APPS[appId]);
      if (onAction) onAction({ type: 'switchApp', appId });
    }
  }), [onAction]);

  // helper to get element id attr
  const getEl = (id) => (el) => register(id, el);

  return (
    <div ref={rootRef} style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', flexDirection: 'column' }}>
      {/* Top bar */}
      <div className="mobile-app-header">
        <div className="mobile-app-title">{app.title}</div>
        <div className="mobile-app-balance">{app.balance || ''}</div>
      </div>

      <div className="mobile-app-content">
        {/* dynamic content */}
        {app.id === 'banking_v1' && (
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
            <button id={app.primaryButton.id} ref={getEl(app.primaryButton.id)} className="mobile-primary" onClick={() => {
              document.getElementById('form-v1').style.display = 'block';
            }}>{app.primaryButton.label}</button>

            <div id="form-v1" style={{ display: 'none', marginTop: 6 }}>
              <input ref={getEl('recipient')} id="recipient" defaultValue={app.fields[0].value} placeholder={app.fields[0].placeholder} className="mobile-input"/>
              <input ref={getEl('amount')} id="amount" defaultValue={app.fields[1].value} placeholder={app.fields[1].placeholder} className="mobile-input"/>
              <button id="confirm" ref={getEl('confirm')} className="mobile-confirm" onClick={() => {
                // show confirmation toast
                const t = document.getElementById('toast-v1');
                t.innerText = `Transaction Complete! ${app.fields[1].value} sent to ${app.fields[0].value}`;
                t.style.display = 'block';
                setTimeout(()=> t.style.display = 'none', 2500);
              }}>Confirm</button>
            </div>
            <div id="toast-v1" className="mobile-toast" style={{ display: 'none' }} />
          </div>
        )}

        {app.id === 'banking_v2' && (
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
            <div style={{ display:'flex', gap:8 }}>
              <button id={app.primaryButton.id} ref={getEl(app.primaryButton.id)} className="mobile-secondary">{app.primaryButton.label}</button>
              <button id="quick-pay" ref={getEl('quick-pay')} className="mobile-secondary">Quick Transfer</button>
            </div>

            <div id="form-v2" style={{ display:'none', marginTop: 6 }}>
              <input id="payee" ref={getEl('payee')} placeholder={app.fields[0].placeholder} className="mobile-input" />
              <input id="amount-v2" ref={getEl('amount-v2')} placeholder={app.fields[1].placeholder} className="mobile-input" />
              <button id="submit" ref={getEl('submit')} className="mobile-confirm" onClick={() => {
                const t = document.getElementById('toast-v2'); t.innerText = 'Payment Complete!'; t.style.display = 'block';
                setTimeout(()=> t.style.display = 'none', 1800);
              }}>Pay Now</button>
            </div>

            <div id="toast-v2" className="mobile-toast" style={{ display: 'none' }} />
          </div>
        )}

        {app.id === 'calendar' && (
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
            {app.events.map(event => (
              <div key={event.id} className="cross-app-event-card">
                <div className="cross-app-event-title">{event.title}</div>
                <div className="cross-app-event-time">{event.time} - {event.location}</div>
              </div>
            ))}
            <button id={app.primaryButton.id} ref={getEl(app.primaryButton.id)} className="mobile-primary" onClick={() => {
              const t = document.getElementById('toast-calendar'); 
              t.innerText = 'Invitation sent to team members!'; 
              t.style.display = 'block';
              setTimeout(()=> t.style.display = 'none', 2000);
            }}>{app.primaryButton.label}</button>
            <div id="toast-calendar" className="mobile-toast" style={{ display: 'none' }} />
          </div>
        )}

        {app.id === 'chat' && (
          <div style={{ display: 'flex', flexDirection:'column', gap:10 }}>
            <div style={{ flex: 1, overflowY: 'auto', padding: 8, background: 'rgba(255,255,255,0.01)', borderRadius: 8 }}>
              {app.messages.map(m => (
                <div key={m.id} style={{
                  marginBottom:8, 
                  color: m.from === 'system' ? '#9fcfe6' : '#d8f7ff',
                  padding: '8px 12px',
                  backgroundColor: m.from === 'system' ? 'rgba(0,229,255,0.1)' : 'rgba(0,255,153,0.1)',
                  borderRadius: '12px',
                  fontSize: '14px'
                }}>
                  {m.text}
                </div>
              ))}
            </div>
            <button id="send-message" ref={getEl('send-message')} className="mobile-secondary" onClick={() => {
              const t = document.getElementById('toast-chat'); 
              t.innerText = 'Message sent successfully!'; 
              t.style.display = 'block';
              setTimeout(()=> t.style.display = 'none', 1500);
            }}>Send Message</button>
            <div id="toast-chat" className="mobile-toast" style={{ display: 'none' }} />
          </div>
        )}

        {app.id === 'email' && (
          <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
            <button id={app.primaryButton.id} ref={getEl(app.primaryButton.id)} className="mobile-primary" onClick={() => {
              document.getElementById('form-email').style.display = 'block';
            }}>{app.primaryButton.label}</button>

            <div id="form-email" style={{ display: 'none', marginTop: 6 }}>
              <input ref={getEl('to')} id="to" defaultValue={app.fields[0].value} placeholder={app.fields[0].placeholder} className="mobile-input"/>
              <input ref={getEl('subject')} id="subject" defaultValue={app.fields[1].value} placeholder={app.fields[1].placeholder} className="mobile-input"/>
              <textarea ref={getEl('body')} id="body" placeholder="Meeting details and agenda..." className="mobile-input" style={{ height: '80px', resize: 'none' }} />
              <button id="send-email" ref={getEl('send-email')} className="mobile-confirm" onClick={() => {
                const t = document.getElementById('toast-email');
                t.innerText = 'Email sent successfully!';
                t.style.display = 'block';
                setTimeout(()=> t.style.display = 'none', 2000);
              }}>Send Email</button>
            </div>
            <div id="toast-email" className="mobile-toast" style={{ display: 'none' }} />
          </div>
        )}
      </div>

      {/* overlays */}
      <GestureOverlay />
      <AttentionHeatmap containerRef={rootRef} elementsRef={elementsRef} attentionData={attentionData} />
      <OCROverlay elementsRef={elementsRef} />
    </div>
  );
});

export default MobileScreen;





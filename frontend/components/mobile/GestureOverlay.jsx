import React, { useEffect, useState } from 'react';
import { AnimatePresence } from 'framer-motion';

let idCounter = 0;

export default function GestureOverlay() {
  const [ripples, setRipples] = useState([]);
  
  useEffect(() => {
    // listen to custom window events for demo triggers
    const handler = (e) => {
      const { type, x, y } = e.detail || {};
      if (type === 'tap') {
        const id = ++idCounter;
        setRipples(r => [...r, { id, x, y }]);
        setTimeout(() => setRipples(r => r.filter(t => t.id !== id)), 900);
      }
      if (type === 'swipe') {
        const id = ++idCounter;
        setRipples(r => [...r, { id, x, y, swipe: true, to: e.detail.to }]);
        setTimeout(() => setRipples(r => r.filter(t => t.id !== id)), 1200);
      }
      if (type === 'type') {
        const id = ++idCounter;
        setRipples(r => [...r, { id, x, y, type: true, text: e.detail.text }]);
        setTimeout(() => setRipples(r => r.filter(t => t.id !== id)), 1500);
      }
    };
    
    window.addEventListener('autorlgesture', handler);
    return () => window.removeEventListener('autorlgesture', handler);
  }, []);

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      <AnimatePresence>
        {ripples.map(r => (
          r.swipe ? (
            <motion.div 
              key={r.id} 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }} 
              style={{
                position: 'absolute', 
                left: r.x - 10, 
                top: r.y - 10
              }}
            >
              {/* simple swipe line */}
              <motion.div 
                initial={{ width: 0 }} 
                animate={{ width: Math.max(80, Math.abs(r.to?.x - r.x || 100)) }} 
                style={{ 
                  height: 6, 
                  background: 'linear-gradient(90deg,#00e3ff,#7a6bff)', 
                  borderRadius: 8 
                }} 
                transition={{ duration: 0.6 }} 
              />
            </motion.div>
          ) : r.type ? (
            <motion.div 
              key={r.id} 
              initial={{ opacity: 0, scale: 0.5 }} 
              animate={{ opacity: 1, scale: 1 }} 
              exit={{ opacity: 0, scale: 0.5 }} 
              style={{
                position: 'absolute', 
                left: r.x - 20, 
                top: r.y - 30,
                background: 'rgba(0,227,255,0.9)',
                color: '#02121a',
                padding: '4px 8px',
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 600,
                border: '1px solid rgba(0,227,255,0.3)'
              }}
              transition={{ duration: 0.3 }}
            >
              "{r.text}"
            </motion.div>
          ) : (
            <motion.div 
              key={r.id} 
              initial={{ scale: 0.2, opacity: 0.7 }} 
              animate={{ scale: 1.8, opacity: 0 }} 
              exit={{ opacity: 0 }} 
              style={{
                position: 'absolute', 
                left: r.x - 28, 
                top: r.y - 28, 
                width: 56, 
                height: 56, 
                borderRadius: 28, 
                border: '2px solid rgba(0,227,255,0.75)',
                boxShadow: '0 0 20px rgba(0,227,255,0.5)'
              }} 
            />
          )
        ))}
      </AnimatePresence>
    </div>
  );
}





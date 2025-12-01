import React, { useEffect, useState } from 'react';

export default function OCROverlay({ elementsRef = { current: {} }, showLabels = true }) {
  const [labels, setLabels] = useState([]);

  useEffect(() => {
    const update = () => {
      const arr = [];
      for (const id in elementsRef.current) {
        const el = elementsRef.current[id];
        if (!el) continue;
        const r = el.getBoundingClientRect();
        arr.push({ 
          id, 
          x: r.left, 
          y: r.top, 
          w: r.width, 
          h: r.height, 
          label: el.id || id, 
          conf: (Math.random() * 0.2 + 0.8).toFixed(2) 
        });
      }
      setLabels(arr);
    };
    
    update();
    const t = setInterval(update, 600);
    return () => clearInterval(t);
  }, [elementsRef]);

  if (!showLabels) return null;

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      {labels.map(l => (
        <div key={l.id} style={{
          position: 'absolute',
          left: l.x,
          top: l.y - 22,
          background: 'rgba(2,12,20,0.8)',
          color: '#eaf6ff',
          padding: '4px 8px',
          borderRadius: 8,
          fontSize: 11,
          border: '1px solid rgba(0,227,255,0.15)',
          backdropFilter: 'blur(4px)',
          whiteSpace: 'nowrap'
        }}>
          {l.label} • {l.conf}
          <div style={{ 
            position:'absolute', 
            left: 0, 
            top: 22, 
            width: l.w, 
            height: l.h, 
            border: '1px dashed rgba(0,227,255,0.2)', 
            borderRadius: 6 
          }} />
        </div>
      ))}
    </div>
  );
}


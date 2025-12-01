import React from 'react';
import './mobile.css';

export default function DeviceSimulator({ children, scale = 0.95, style = {} }) {
  // Device chrome and background; children should be the MobileScreen
  return (
    <div className="device-shell" style={{ transform: `scale(${scale})`, ...style }}>
      <div className="device-bezel">
        <div className="device-notch" />
        <div className="device-screen">
          {children}
        </div>
        <div className="device-home" />
      </div>
    </div>
  );
}





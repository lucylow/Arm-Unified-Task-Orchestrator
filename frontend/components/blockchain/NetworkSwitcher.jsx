import React, { useState } from 'react';
import { useWeb3 } from '../../contexts/Web3Context';

export default function NetworkSwitcher() {
  const { chainId, getNetworkName, switchNetwork, supportedChains, isConnected } = useWeb3();
  const [isOpen, setIsOpen] = useState(false);
  const [switching, setSwitching] = useState(false);

  const handleNetworkSwitch = async (targetChainId) => {
    setSwitching(true);
    try {
      await switchNetwork(targetChainId);
      setIsOpen(false);
    } catch (error) {
      console.error('Network switch error:', error);
    } finally {
      setSwitching(false);
    }
  };

  if (!isConnected) return null;

  return (
    <div style={{ position: 'relative' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          padding: '8px 12px',
          borderRadius: 6,
          background: '#f0f0f0',
          border: '1px solid #ddd',
          cursor: 'pointer',
          fontWeight: 500,
          fontSize: 13,
          display: 'flex',
          alignItems: 'center',
          gap: 6
        }}
      >
        <span style={{ 
          width: 8, 
          height: 8, 
          borderRadius: '50%', 
          background: '#00e676',
          display: 'inline-block'
        }}></span>
        {getNetworkName()}
        <span style={{ fontSize: 10 }}>▼</span>
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          right: 0,
          marginTop: 4,
          background: 'white',
          border: '1px solid #ddd',
          borderRadius: 8,
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          minWidth: 200,
          zIndex: 1000
        }}>
          <div style={{ padding: 8, borderBottom: '1px solid #eee', fontWeight: 600, fontSize: 12, color: '#666' }}>
            Switch Network
          </div>
          {Object.entries(supportedChains).map(([id, config]) => {
            const numId = parseInt(id);
            const isCurrentChain = numId === chainId;
            
            return (
              <button
                key={id}
                onClick={() => !isCurrentChain && handleNetworkSwitch(numId)}
                disabled={switching || isCurrentChain}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  border: 'none',
                  background: isCurrentChain ? '#e8f5e9' : 'white',
                  cursor: isCurrentChain ? 'default' : 'pointer',
                  textAlign: 'left',
                  fontSize: 13,
                  borderBottom: '1px solid #f0f0f0',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
                onMouseOver={(e) => {
                  if (!isCurrentChain) e.currentTarget.style.background = '#f5f5f5';
                }}
                onMouseOut={(e) => {
                  if (!isCurrentChain) e.currentTarget.style.background = 'white';
                }}
              >
                <span>{config.name}</span>
                {isCurrentChain && (
                  <span style={{ color: '#00e676', fontSize: 12 }}>✓ Active</span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}


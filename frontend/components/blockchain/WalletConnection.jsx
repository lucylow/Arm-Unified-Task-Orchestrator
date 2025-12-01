import React from 'react';
import { useWeb3 } from '../../contexts/Web3Context';
import NetworkSwitcher from './NetworkSwitcher';

export default function WalletConnection(){
  const { account, isConnected, connect, disconnect, error, isMetaMaskInstalled, balance, getNetworkName, mockMode, toggleMockMode } = useWeb3();
  
  return (
    <div style={{display:'flex',flexDirection:'column',gap:8,alignItems:'flex-start',maxWidth:400}}>
      {mockMode && (
        <div style={{
          color:'#1976d2',
          background:'#e3f2fd',
          padding:12,
          borderRadius:8,
          fontSize:14,
          border:'1px solid #90caf9',
          width:'100%'
        }}>
          <strong>🎭 Demo Mode Active</strong>
          <p style={{margin:'8px 0 0 0',fontSize:13}}>
            Using mock wallet data. All blockchain features are simulated.
            {isMetaMaskInstalled && (
              <button
                onClick={toggleMockMode}
                style={{
                  marginLeft: 8,
                  padding: '4px 8px',
                  fontSize: 12,
                  borderRadius: 4,
                  border: 'none',
                  background: '#1976d2',
                  color: 'white',
                  cursor: 'pointer'
                }}
              >
                Switch to Real Wallet
              </button>
            )}
          </p>
        </div>
      )}

      {error && (
        <div style={{
          color:'#ff4444',
          background:'#ffeeee',
          padding:12,
          borderRadius:8,
          fontSize:14,
          border:'1px solid #ffcccc',
          width:'100%'
        }}>
          <strong>⚠️ Error:</strong> {error}
        </div>
      )}
      
      {!isMetaMaskInstalled && !mockMode && (
        <div style={{
          color:'#ff9800',
          background:'#fff3e0',
          padding:12,
          borderRadius:8,
          fontSize:14,
          border:'1px solid #ffe0b2',
          width:'100%'
        }}>
          <strong>📦 MetaMask Not Found</strong>
          <p style={{margin:'8px 0 0 0'}}>
            Please install MetaMask extension to use blockchain features.
            <a 
              href="https://metamask.io/download/" 
              target="_blank" 
              rel="noopener noreferrer"
              style={{
                display:'block',
                marginTop:8,
                color:'#ff9800',
                textDecoration:'underline'
              }}
            >
              Download MetaMask →
            </a>
          </p>
        </div>
      )}
      
      <div style={{display:'flex',gap:8,alignItems:'center',width:'100%',flexWrap:'wrap'}}>
        {!isConnected ? (
          <button 
            onClick={connect} 
            style={{
              padding:'10px 20px',
              borderRadius:8,
              background: mockMode ? '#1976d2' : (isMetaMaskInstalled ? '#00e3ff' : '#cccccc'),
              color: mockMode || isMetaMaskInstalled ? '#fff' : '#666',
              border:'none',
              fontWeight:600,
              cursor: 'pointer',
              transition:'all 0.2s',
              fontSize:15
            }}
            onMouseOver={(e) => {
              if (mockMode) {
                e.target.style.background = '#1565c0';
              } else if (isMetaMaskInstalled) {
                e.target.style.background = '#00c8e0';
              }
            }}
            onMouseOut={(e) => {
              if (mockMode) {
                e.target.style.background = '#1976d2';
              } else if (isMetaMaskInstalled) {
                e.target.style.background = '#00e3ff';
              }
            }}
          >
            {mockMode ? '🎭 Connect Mock Wallet' : (isMetaMaskInstalled ? '🦊 Connect Wallet' : 'Install MetaMask First')}
          </button>
        ) : (
          <>
            <div style={{
              display:'flex',
              gap:12,
              alignItems:'center',
              background:'#f0f0f0',
              padding:'8px 16px',
              borderRadius:8,
              flex:1,
              justifyContent:'space-between'
            }}>
              <div style={{display:'flex',flexDirection:'column',gap:4}}>
                <div style={{fontWeight:700,fontSize:14}}>
                  {String(account).slice(0,8)}...{String(account).slice(-6)}
                </div>
                <div style={{display:'flex',gap:8,alignItems:'center'}}>
                  {balance && (
                    <div style={{fontSize:12,color:'#666'}}>
                      {parseFloat(balance).toFixed(4)} ETH
                    </div>
                  )}
                  <div style={{fontSize:11,color:'#999'}}>
                    • {getNetworkName()}
                  </div>
                </div>
              </div>
              <button 
                onClick={disconnect} 
                style={{
                  padding:'6px 12px',
                  borderRadius:6,
                  background:'#ff4444',
                  color:'white',
                  border:'none',
                  cursor:'pointer',
                  fontWeight:500,
                  fontSize:13
                }}
                onMouseOver={(e) => e.target.style.background = '#cc0000'}
                onMouseOut={(e) => e.target.style.background = '#ff4444'}
              >
                Disconnect
              </button>
            </div>
            <NetworkSwitcher />
          </>
        )}
      </div>
    </div>
  );
}

import React from 'react';
import WalletConnection from '../components/blockchain/WalletConnection';
import AchievementSystem from '../components/blockchain/AchievementSystem';
import TaskVerification from '../components/blockchain/TaskVerification';

export default function BlockchainEnhancedDashboard(){
  return (
    <div style={{padding:20}}>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
        <h1>Blockchain Enhanced Dashboard</h1>
        <WalletConnection />
      </div>
      <div style={{display:'grid',gridTemplateColumns:'2fr 1fr',gap:20,marginTop:20}}>
        <div>
          <AchievementSystem />
          <div style={{height:12}} />
          <TaskVerification />
        </div>
        <div>
          <div style={{padding:12,background:'#fff',borderRadius:8}}>Blockchain Stats (mock)</div>
        </div>
      </div>
    </div>
  );
}

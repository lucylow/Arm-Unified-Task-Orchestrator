import React, { useState } from 'react';
import { useWeb3 } from '../../contexts/Web3Context';
import { MOCK_ACHIEVEMENTS } from '../../data/mock-blockchain-data';

export default function AchievementSystem(){
  const { isConnected } = useWeb3();
  const [unlocked] = useState(MOCK_ACHIEVEMENTS.readyToMint || []);

  const mint = async (a) => {
    // mock mint
    alert('Mock mint: ' + a.name);
  };

  if(!isConnected){
    return <div style={{padding:12,background:'#fff',color:'#000',borderRadius:8}}>Connect wallet to mint achievements</div>;
  }

  return (
    <div style={{padding:12,background:'#fff',color:'#000',borderRadius:8}}>
      <h3>Achievements</h3>
      {unlocked.map(u=> (
        <div key={u.id} style={{display:'flex',justifyContent:'space-between',alignItems:'center',padding:8,borderBottom:'1px solid #eee'}}>
          <div>
            <div style={{fontWeight:700}}>{u.name}</div>
            <div style={{fontSize:12,color:'#666'}}>{u.description}</div>
          </div>
          <button onClick={()=>mint(u)}>Mint</button>
        </div>
      ))}
    </div>
  );
}

import React from 'react';
import { useWeb3 } from '../../contexts/Web3Context';
import { MOCK_VERIFIED_TASKS } from '../../data/mock-blockchain-data';

export default function TaskVerification(){
  const { isConnected } = useWeb3();
  if(!isConnected) return null;
  return (
    <div style={{padding:12,background:'#fff',borderRadius:8}}>
      <h3>On-chain Verified Tasks</h3>
      <div>No verified tasks in mock data.</div>
    </div>
  );
}

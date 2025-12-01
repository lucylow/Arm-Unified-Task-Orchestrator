import React, { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, HashRouter } from 'react-router-dom';
import './index.css';
import App from './App';

// Error overlay: if runtime errors occur (white page), show a helpful message
// But don't show for non-blocking errors like MetaMask issues
function showFatalError(err){
  try{
    const errString = String(err);
    
    // Don't show fatal error for MetaMask or wallet-related issues
    if (errString.includes('MetaMask') || 
        errString.includes('ethereum') || 
        errString.includes('wallet') ||
        errString.includes('Web3')) {
      console.warn('Non-fatal wallet error (app will continue):', err);
      return;
    }

    const root = document.getElementById('root');
    if(root && root.innerHTML.trim() === ''){
      root.innerHTML = `
        <div style="font-family:Inter,Arial,sans-serif;padding:24px;color:#041522;background:#fff;">
          <h2 style="color:#b00202;">Application Error</h2>
          <pre style="white-space:pre-wrap;color:#101010;">${errString}</pre>
          <p>Please open the browser console for full stack trace.</p>
        </div>
      `;
    }
  }catch(e){
    console.error('Failed to render fatal error', e);
  }
}

window.addEventListener('error', (ev) => {
  console.error('Global error caught', ev.error || ev.message);
  // Only show fatal error for critical issues
  const errString = String(ev.error || ev.message);
  if (!errString.includes('MetaMask') && !errString.includes('ethereum') && !errString.includes('Web3')) {
    showFatalError(ev.error || ev.message || 'Unknown error');
  }
});
window.addEventListener('unhandledrejection', (ev) => {
  console.error('Unhandled rejection', ev.reason);
  // Only show fatal error for critical promise rejections
  const errString = String(ev.reason);
  if (!errString.includes('MetaMask') && !errString.includes('ethereum') && !errString.includes('Web3')) {
    showFatalError(ev.reason || 'Unhandled promise rejection');
  }
});

// Some static hosts don't rewrite unknown paths to index.html. Fall back to
// HashRouter when served from a nested path or common preview hostnames.
const hostname = (typeof window !== 'undefined' && window.location && window.location.hostname) ? window.location.hostname : '';
const pathname = (typeof window !== 'undefined' && window.location && window.location.pathname) ? window.location.pathname : '/';
const isStaticHost = hostname.includes('lovable') || hostname.endsWith('github.io') || pathname.startsWith('/projects/');
const Router = isStaticHost ? HashRouter : BrowserRouter;

// Simple ErrorBoundary so React render errors are visible
class ErrorBoundary extends React.Component {
  constructor(props){ super(props); this.state = {error:null}; }
  static getDerivedStateFromError(error){ return {error}; }
  componentDidCatch(error, info){ console.error('React ErrorBoundary:', error, info); }
  render(){
    if(this.state.error){
      return (
        <div style={{fontFamily:'Inter,Arial,sans-serif',padding:24}}>
          <h2 style={{color:'#b00202'}}>Application Error</h2>
          <pre style={{whiteSpace:'pre-wrap'}}>{String(this.state.error)}</pre>
          <p>Open the browser console for more details.</p>
        </div>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <Router>
        <App />
      </Router>
    </ErrorBoundary>
  </StrictMode>
);

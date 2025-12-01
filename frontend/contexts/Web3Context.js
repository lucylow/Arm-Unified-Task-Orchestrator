import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { ethers } from 'ethers';
import { CONTRACT_CONFIG, DEFAULT_CHAIN_ID, SUPPORTED_CHAINS } from '../blockchain/web3-config';

const Web3Context = createContext(null);
export const useWeb3 = () => useContext(Web3Context);

// Mock mode: Enable by setting VITE_MOCK_WEB3=true in .env or when MetaMask is not available
const ENABLE_MOCK_MODE = import.meta.env.VITE_MOCK_WEB3 === 'true';

// Mock wallet data for demo/development
const MOCK_WALLET = {
  account: '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
  balance: '12.5',
  chainId: 1,
  chainName: 'Ethereum Mainnet',
};

export const Web3Provider = ({ children }) => {
  const [provider, setProvider] = useState(null);
  const [signer, setSigner] = useState(null);
  const [account, setAccount] = useState(null);
  const [chainId, setChainId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [contracts, setContracts] = useState({});
  const [balance, setBalance] = useState('0');
  const [error, setError] = useState(null);
  const [isMetaMaskInstalled, setIsMetaMaskInstalled] = useState(false);
  const [mockMode, setMockMode] = useState(ENABLE_MOCK_MODE);

  // Check if MetaMask is installed
  useEffect(() => {
    // Don't check for MetaMask in mock mode
    if (mockMode) {
      console.log('🎭 Web3 running in MOCK MODE - MetaMask not required');
      setIsMetaMaskInstalled(false);
      return;
    }

    const checkMetaMask = () => {
      try {
        if (typeof window !== 'undefined' && typeof window.ethereum !== 'undefined') {
          setIsMetaMaskInstalled(true);
          console.log('✅ MetaMask detected');
          setError(null);
        } else {
          setIsMetaMaskInstalled(false);
          console.log('ℹ️ MetaMask not detected - app will work with mock data');
          // Enable mock mode if MetaMask is not available
          if (!mockMode) {
            setMockMode(true);
            console.log('🎭 Automatically enabled mock mode');
          }
        }
      } catch (e) {
        console.warn('Error checking MetaMask:', e);
        setIsMetaMaskInstalled(false);
        if (!mockMode) {
          setMockMode(true);
          console.log('🎭 Enabled mock mode due to error');
        }
      }
    };

    checkMetaMask();

    // Listen for MetaMask installation
    if (typeof window !== 'undefined') {
      window.addEventListener('ethereum#initialized', checkMetaMask, { once: true });

      // Backup check after 3 seconds
      setTimeout(checkMetaMask, 3000);

      return () => {
        window.removeEventListener('ethereum#initialized', checkMetaMask);
      };
    }
  }, [mockMode]);

  const checkAccounts = useCallback(async () => {
    if (typeof window === 'undefined' || !window.ethereum || !provider || mockMode) return;
    
    try {
      const accs = await window.ethereum.request({ method: 'eth_accounts' });
      if (accs && accs.length) {
        const s = await provider.getSigner();
        setSigner(s);
        setAccount(accs[0]);
        const net = await provider.getNetwork();
        setChainId(Number(net.chainId));
        setIsConnected(true);
        setError(null);
        
        // Get balance
        try {
          const bal = await provider.getBalance(accs[0]);
          setBalance(ethers.formatEther(bal));
        } catch (e) {
          console.warn('Failed to get balance:', e);
        }
        
        // Initialize contracts
        const created = {};
        Object.keys(CONTRACT_CONFIG).forEach(k => {
          const cfg = CONTRACT_CONFIG[k];
          if (cfg.abi && cfg.address && cfg.address !== '0x0000000000000000000000000000000000000000') {
            try { 
              created[k] = new ethers.Contract(cfg.address, cfg.abi, s);
            } catch(e) { 
              console.warn('Failed to create contract:', k, e); 
            }
          }
        });
        setContracts(created);
      }
    } catch (e) {
      console.warn('checkAccounts error:', e);
    }
  }, [provider, mockMode]);

  useEffect(() => {
    // Skip MetaMask initialization in mock mode
    if (mockMode) return;

    if (typeof window !== 'undefined' && window.ethereum && isMetaMaskInstalled) {
      try {
        const p = new ethers.BrowserProvider(window.ethereum);
        setProvider(p);
        checkAccounts();

        // Set up event listeners
        const handleAccountsChanged = (accounts) => {
          console.log('Accounts changed:', accounts);
          if (accounts.length === 0) {
            disconnect();
          } else {
            checkAccounts();
          }
        };

        const handleChainChanged = (chainIdHex) => {
          console.log('Chain changed to:', chainIdHex);
          window.location.reload();
        };

        const handleDisconnect = () => {
          console.log('MetaMask disconnected');
          disconnect();
        };

        window.ethereum.on('accountsChanged', handleAccountsChanged);
        window.ethereum.on('chainChanged', handleChainChanged);
        window.ethereum.on('disconnect', handleDisconnect);

        return () => {
          if (window.ethereum && window.ethereum.removeListener) {
            window.ethereum.removeListener('accountsChanged', handleAccountsChanged);
            window.ethereum.removeListener('chainChanged', handleChainChanged);
            window.ethereum.removeListener('disconnect', handleDisconnect);
          }
        };
      } catch (e) {
        console.error('Failed to initialize Web3 provider:', e);
      }
    }
  }, [checkAccounts, isMetaMaskInstalled, mockMode]);

  const connect = async () => {
    setError(null);

    // Mock mode connection
    if (mockMode) {
      console.log('🎭 Connecting to mock wallet...');
      setAccount(MOCK_WALLET.account);
      setBalance(MOCK_WALLET.balance);
      setChainId(MOCK_WALLET.chainId);
      setIsConnected(true);
      console.log('✅ Mock wallet connected:', MOCK_WALLET.account);
      return;
    }

    if (typeof window === 'undefined' || !window.ethereum) {
      const errorMsg = 'MetaMask is not installed. Switching to mock mode...';
      console.log(errorMsg);
      setMockMode(true);
      // Auto-connect in mock mode
      setTimeout(() => connect(), 100);
      return;
    }

    try {
      console.log('Requesting MetaMask connection...');
      const accs = await window.ethereum.request({ method: 'eth_requestAccounts' });
      
      if (!accs || accs.length === 0) {
        throw new Error('No accounts returned from MetaMask');
      }

      console.log('Connected to account:', accs[0]);
      
      const p = new ethers.BrowserProvider(window.ethereum);
      setProvider(p);
      const s = await p.getSigner();
      setSigner(s);
      setAccount(accs[0]);
      const net = await p.getNetwork();
      setChainId(Number(net.chainId));
      setIsConnected(true);

      // Get balance
      try {
        const bal = await p.getBalance(accs[0]);
        setBalance(ethers.formatEther(bal));
      } catch (e) {
        console.warn('Failed to get balance:', e);
      }

      // Create contract instances if ABI present
      const created = {};
      Object.keys(CONTRACT_CONFIG).forEach(k => {
        const cfg = CONTRACT_CONFIG[k];
        if (cfg.abi && cfg.address && cfg.address !== '0x0000000000000000000000000000000000000000') {
          try { 
            created[k] = new ethers.Contract(cfg.address, cfg.abi, s);
            console.log(`Contract ${k} initialized`);
          } catch(e) { 
            console.warn('Failed to create contract:', k, e); 
          }
        }
      });
      setContracts(created);

      console.log('MetaMask connected successfully!');
      setError(null);
      
    } catch (e) {
      console.error('MetaMask connection error:', e);
      
      let errorMessage = 'Failed to connect to MetaMask';
      
      if (e.code === 4001) {
        errorMessage = 'Connection request rejected. Please approve the connection in MetaMask.';
      } else if (e.code === -32002) {
        errorMessage = 'Connection request pending. Please check MetaMask extension.';
      } else if (e.message) {
        errorMessage = e.message;
      }
      
      setError(errorMessage);
    }
  };

  const disconnect = () => {
    console.log('Disconnecting wallet...');
    setProvider(null);
    setSigner(null);
    setAccount(null);
    setChainId(null);
    setIsConnected(false);
    setContracts({});
    setBalance('0');
    setError(null);
  };

  const switchNetwork = async (targetChainId) => {
    // Mock mode network switching
    if (mockMode) {
      console.log(`🎭 Mock: Switched to chain ${targetChainId}`);
      setChainId(targetChainId);
      return true;
    }

    if (typeof window === 'undefined' || !window.ethereum) {
      setError('MetaMask is not installed');
      return false;
    }

    try {
      const chainIdHex = '0x' + targetChainId.toString(16);
      
      await window.ethereum.request({
        method: 'wallet_switchEthereumChain',
        params: [{ chainId: chainIdHex }],
      });
      
      console.log(`Switched to chain ${targetChainId}`);
      return true;
    } catch (switchError) {
      if (switchError.code === 4902) {
        try {
          const chainConfig = SUPPORTED_CHAINS[targetChainId];
          if (!chainConfig) {
            throw new Error('Chain not supported');
          }

          await window.ethereum.request({
            method: 'wallet_addEthereumChain',
            params: [{
              chainId: '0x' + targetChainId.toString(16),
              chainName: chainConfig.name,
              rpcUrls: [chainConfig.rpc],
              blockExplorerUrls: chainConfig.explorer ? [chainConfig.explorer] : [],
            }],
          });
          
          return true;
        } catch (addError) {
          console.error('Error adding network:', addError);
          setError('Failed to add network: ' + addError.message);
          return false;
        }
      } else {
        console.error('Error switching network:', switchError);
        setError('Failed to switch network: ' + switchError.message);
        return false;
      }
    }
  };

  const getNetworkName = () => {
    if (!chainId) return mockMode ? 'Mock Network (Demo Mode)' : 'Not connected';
    return SUPPORTED_CHAINS[chainId]?.name || `Unknown (${chainId})`;
  };

  const toggleMockMode = () => {
    const newMockMode = !mockMode;
    setMockMode(newMockMode);
    console.log(`🎭 Mock mode ${newMockMode ? 'enabled' : 'disabled'}`);
    
    // Reset connection state when toggling
    disconnect();
  };

  return (
    <Web3Context.Provider value={{ 
      provider, 
      signer, 
      account, 
      chainId, 
      isConnected, 
      contracts, 
      balance, 
      error, 
      connect, 
      disconnect,
      switchNetwork,
      getNetworkName,
      isMetaMaskInstalled,
      supportedChains: SUPPORTED_CHAINS,
      mockMode,
      toggleMockMode
    }}>
      {children}
    </Web3Context.Provider>
  );
};

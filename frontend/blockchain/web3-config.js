/* eslint-disable no-undef */

// Sample ERC721 ABI (for Achievement NFTs)
const ERC721_ABI = [
  "function mint(address to, uint256 tokenId) external",
  "function balanceOf(address owner) view returns (uint256)",
  "function tokenURI(uint256 tokenId) view returns (string)",
  "event Transfer(address indexed from, address indexed to, uint256 indexed tokenId)"
];

// Sample ERC20 ABI (for Governance Token)
const ERC20_ABI = [
  "function balanceOf(address account) view returns (uint256)",
  "function transfer(address to, uint256 amount) returns (bool)",
  "function allowance(address owner, address spender) view returns (uint256)",
  "function approve(address spender, uint256 amount) returns (bool)",
  "event Transfer(address indexed from, address indexed to, uint256 value)"
];

// Sample Task Verification Contract ABI
const TASK_VERIFICATION_ABI = [
  "function verifyTask(string taskId, address user) external",
  "function isTaskVerified(string taskId, address user) view returns (bool)",
  "function getVerifiedTasks(address user) view returns (string[])",
  "event TaskVerified(string indexed taskId, address indexed user, uint256 timestamp)"
];

export const CONTRACT_CONFIG = {
  ACHIEVEMENT_NFT: {
    address: (typeof process !== 'undefined' && process.env.REACT_APP_NFT_CONTRACT_ADDRESS) || '0x0000000000000000000000000000000000000000',
    abi: ERC721_ABI
  },
  TASK_VERIFICATION: {
    address: (typeof process !== 'undefined' && process.env.REACT_APP_TASK_CONTRACT_ADDRESS) || '0x0000000000000000000000000000000000000000',
    abi: TASK_VERIFICATION_ABI
  },
  GOVERNANCE_TOKEN: {
    address: (typeof process !== 'undefined' && process.env.REACT_APP_TOKEN_CONTRACT_ADDRESS) || '0x0000000000000000000000000000000000000000',
    abi: ERC20_ABI
  }
};

// Supported networks
export const SUPPORTED_CHAINS = {
  1: { name: 'Ethereum Mainnet', rpc: 'https://mainnet.infura.io/v3/', explorer: 'https://etherscan.io' },
  5: { name: 'Goerli Testnet', rpc: 'https://goerli.infura.io/v3/', explorer: 'https://goerli.etherscan.io' },
  11155111: { name: 'Sepolia Testnet', rpc: 'https://sepolia.infura.io/v3/', explorer: 'https://sepolia.etherscan.io' },
  137: { name: 'Polygon Mainnet', rpc: 'https://polygon-rpc.com/', explorer: 'https://polygonscan.com' },
  80001: { name: 'Mumbai Testnet', rpc: 'https://rpc-mumbai.maticvigil.com/', explorer: 'https://mumbai.polygonscan.com' },
  31337: { name: 'Hardhat Local', rpc: 'http://127.0.0.1:8545/', explorer: '' }
};

export const DEFAULT_CHAIN_ID = (typeof process !== 'undefined' && process.env.REACT_APP_DEFAULT_CHAIN_ID && parseInt(process.env.REACT_APP_DEFAULT_CHAIN_ID)) || 11155111;

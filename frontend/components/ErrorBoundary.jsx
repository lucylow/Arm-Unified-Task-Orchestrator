import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
  }

  render() {
    if (this.state.hasError) {
      const errorString = this.state.error ? this.state.error.toString() : '';
      const isWeb3Error = errorString.includes('MetaMask') || errorString.includes('ethers') || errorString.includes('Web3') || errorString.includes('blockchain');
      
      // Fallback UI
      return (
        <div style={{
          padding: '20px',
          margin: '20px',
          border: '1px solid #ff6b6b',
          borderRadius: '8px',
          backgroundColor: '#ffe0e0',
          color: '#d63031',
          fontFamily: 'Arial, sans-serif'
        }}>
          <h2 style={{ margin: '0 0 10px 0', fontSize: '18px' }}>
            {isWeb3Error ? '🔗 Blockchain Feature Error' : '⚠️ Something went wrong'}
          </h2>
          <p style={{ margin: '0 0 15px 0', fontSize: '14px' }}>
            {isWeb3Error ? (
              <>
                This error is related to blockchain functionality. The app will work fine without MetaMask. 
                Blockchain features are optional and not required for core functionality.
              </>
            ) : (
              'An error occurred while loading this component.'
            )}
          </p>
          <details style={{ fontSize: '12px', marginTop: '10px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold' }}>Error Details</summary>
            <pre style={{ 
              marginTop: '10px', 
              padding: '10px', 
              backgroundColor: '#f8f8f8', 
              borderRadius: '4px',
              overflow: 'auto',
              fontSize: '11px'
            }}>
              {this.state.error && this.state.error.toString()}
              <br />
              {this.state.errorInfo && this.state.errorInfo.componentStack}
            </pre>
          </details>
          <button 
            onClick={() => window.location.reload()} 
            style={{
              marginTop: '15px',
              padding: '8px 16px',
              backgroundColor: '#00b894',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
          >
            Reload Page
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

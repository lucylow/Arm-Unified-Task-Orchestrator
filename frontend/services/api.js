/**
 * API Service - Centralized API calls to AutoRL backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000/api';

console.log('🔧 API Service initialized with base URL:', API_BASE_URL);

class ApiService {
  constructor(baseUrl = API_BASE_URL) {
    this.baseUrl = baseUrl;
    console.log('📡 ApiService created with baseUrl:', this.baseUrl);
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    console.log(`📤 API Request: ${options.method || 'GET'} ${url}`);

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        console.error(`❌ API Error [${endpoint}]:`, response.status, error);
        throw new Error(error.message || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`📥 API Response [${endpoint}]:`, data);
      return data;
    } catch (error) {
      console.error(`❌ API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  // Health Check
  async healthCheck() {
    return this.request('/health');
  }

  // Device APIs
  async getDevices() {
    return this.request('/devices');
  }

  // Task APIs
  async getTasks() {
    return this.request('/tasks');
  }

  async createTask(instruction, deviceId = null, parameters = {}) {
    return this.request('/tasks', {
      method: 'POST',
      body: JSON.stringify({
        instruction,
        device_id: deviceId,
        parameters,
      }),
    });
  }

  // Metrics APIs
  async getMetrics() {
    return this.request('/metrics');
  }

  // Activity Log APIs
  async getActivity() {
    return this.request('/activity');
  }

  // Policy APIs
  async getPolicies() {
    return this.request('/policies');
  }

  async promotePolicy(policyName) {
    return this.request(`/policies/promote?policy_name=${policyName}`, {
      method: 'POST',
    });
  }

  // Plugin APIs
  async getPlugins() {
    return this.request('/plugins');
  }

  async executePlugin(pluginName, inputData) {
    return this.request(`/plugins/${pluginName}/execute`, {
      method: 'POST',
      body: JSON.stringify(inputData),
    });
  }

  // Agent Control APIs
  async startAgent() {
    return this.request('/agent/start', {
      method: 'POST',
    });
  }

  async stopAgent() {
    return this.request('/agent/stop', {
      method: 'POST',
    });
  }
}

// Export singleton instance
export const apiService = new ApiService();
export default apiService;


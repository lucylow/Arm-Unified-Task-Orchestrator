import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for fetching and managing analytics data
 * @param {string} initialTimeRange - Initial time range (24h, 7d, 30d, 90d)
 * @returns {Object} Analytics data and helper functions
 */
export function useAnalytics(initialTimeRange = '7d') {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timeRange, setTimeRange] = useState(initialTimeRange);

  /**
   * Generate fallback mock data when API is unavailable
   */
  const generateFallbackData = useCallback(() => {
    const mockData = {
      stats: {
        totalTasks: 1247,
        tasksChange: 18,
        successRate: 94.2,
        successRateChange: 2.1,
        avgDuration: 23.4,
        durationChange: -3.2,
        activeUsers: 42,
        usersChange: 7,
        activeDevices: 8,
        devicesChange: 2,
        totalErrors: 73,
        errorsChange: -12
      },
      taskTrends: generateTaskTrends(),
      deviceUtilization: generateDeviceUtilization(),
      taskDistribution: generateTaskDistribution(),
      performanceMetrics: generatePerformanceMetrics(),
      rlTraining: generateRLTrainingData(),
      errorAnalysis: generateErrorAnalysis()
    };
    return mockData;
  }, []);

  const generateTaskTrends = () => {
    const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
    return days.map(day => ({
      name: day,
      successful: Math.floor(Math.random() * 100) + 50,
      failed: Math.floor(Math.random() * 20),
      pending: Math.floor(Math.random() * 30)
    }));
  };

  const generateDeviceUtilization = () => [
    { name: 'Pixel 6', value: 25, status: 'active' },
    { name: 'iPhone 13', value: 20, status: 'active' },
    { name: 'Galaxy S21', value: 18, status: 'active' },
    { name: 'OnePlus 9', value: 15, status: 'idle' },
    { name: 'Others', value: 22, status: 'idle' }
  ];

  const generateTaskDistribution = () => [
    { name: 'UI Testing', value: 30 },
    { name: 'Data Entry', value: 25 },
    { name: 'Navigation', value: 20 },
    { name: 'Form Filling', value: 15 },
    { name: 'Others', value: 10 }
  ];

  const generatePerformanceMetrics = () => {
    return Array.from({ length: 24 }, (_, i) => ({
      hour: `${i}:00`,
      responseTime: Math.floor(Math.random() * 500) + 200,
      throughput: Math.floor(Math.random() * 100) + 50,
      cpuUsage: Math.floor(Math.random() * 60) + 20
    }));
  };

  const generateRLTrainingData = () => {
    return Array.from({ length: 50 }, (_, i) => ({
      episode: i + 1,
      reward: Math.random() * 100 + (i * 2),
      loss: Math.max(100 - (i * 2), 10) + Math.random() * 20
    }));
  };

  const generateErrorAnalysis = () => [
    { type: 'Timeout', count: 25, percentage: 34 },
    { type: 'Network', count: 18, percentage: 25 },
    { type: 'Element Not Found', count: 15, percentage: 21 },
    { type: 'Device Disconnected', count: 10, percentage: 14 },
    { type: 'Other', count: 5, percentage: 6 }
  ];

  /**
   * Fetch analytics data from the backend
   */
  const fetchAnalytics = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Use relative URL to leverage Vite proxy (maps /api to http://localhost:5000)
      const response = await fetch(`/api/analytics?range=${timeRange}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch analytics: ${response.statusText}`);
      }

      const analyticsData = await response.json();
      setData(analyticsData);
    } catch (err) {
      console.error('Error fetching analytics:', err);
      setError(err.message);
      
      // Fallback to mock data if API fails
      setData(generateFallbackData());
    } finally {
      setIsLoading(false);
    }
  }, [timeRange, generateFallbackData]);

  /**
   * Refresh analytics data
   */
  const refresh = useCallback(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  /**
   * Update time range and fetch new data
   */
  const updateTimeRange = useCallback((newRange) => {
    setTimeRange(newRange);
  }, []);

  /**
   * Export analytics data as CSV
   */
  const exportToCSV = useCallback(() => {
    if (!data) return;

    let csv = 'Analytics Export\n\n';
    
    // Export stats
    csv += 'Key Metrics\n';
    csv += 'Metric,Value,Change\n';
    csv += `Total Tasks,${data.stats.totalTasks},+${data.stats.tasksChange}%\n`;
    csv += `Success Rate,${data.stats.successRate}%,+${data.stats.successRateChange}%\n`;
    csv += `Avg Duration,${data.stats.avgDuration}s,${data.stats.durationChange}s\n`;
    csv += `Active Devices,${data.stats.activeDevices},+${data.stats.devicesChange}\n\n`;

    // Export task trends
    csv += 'Task Trends\n';
    csv += 'Day,Successful,Failed,Pending\n';
    data.taskTrends.forEach(row => {
      csv += `${row.name},${row.successful},${row.failed},${row.pending}\n`;
    });
    csv += '\n';

    // Export device utilization
    csv += 'Device Utilization\n';
    csv += 'Device,Tasks,Status\n';
    data.deviceUtilization.forEach(device => {
      csv += `${device.name},${device.value},${device.status}\n`;
    });
    csv += '\n';

    // Export task distribution
    csv += 'Task Distribution\n';
    csv += 'Task Type,Count\n';
    data.taskDistribution.forEach(task => {
      csv += `${task.name},${task.value}\n`;
    });

    // Create and download CSV file
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analytics-${timeRange}-${new Date().toISOString()}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, [data, timeRange]);

  /**
   * Export analytics data as JSON
   */
  const exportToJSON = useCallback(() => {
    if (!data) return;

    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `analytics-${timeRange}-${new Date().toISOString()}.json`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, [data, timeRange]);

  // Fetch data on mount and when time range changes
  useEffect(() => {
    fetchAnalytics();
  }, [fetchAnalytics]);

  return {
    data,
    isLoading,
    error,
    timeRange,
    refresh,
    updateTimeRange,
    exportToCSV,
    exportToJSON
  };
}

export default useAnalytics;


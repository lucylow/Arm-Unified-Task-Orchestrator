import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CheckCircle, Clock, Smartphone, Zap, Activity, Brain, Eye, Target, Cpu } from 'lucide-react'
import { MetricCard } from '@/components/MetricCard'
import { TaskExecutionView } from '@/components/TaskExecutionView'
import { DeviceStatusPanel } from '@/components/ui/DeviceStatusPanel'
import { EmptyState } from '@/components/EmptyState'
import { DashboardSkeleton } from '@/components/ui/DashboardSkeleton'
import { TaskControlPanel } from '@/components/TaskControlPanel'
import LiveAgentActivity from '@/components/dashboard/LiveAgentActivity'
import PerformanceCharts from '@/components/dashboard/PerformanceCharts'
import { apiService } from '@/services/api'
import { useWebSocket } from '@/hooks/useWebSocket'

export default function Dashboard() {
  const [systemStatus, setSystemStatus] = useState('active')
  const [devices, setDevices] = useState([])
  const [currentTask, setCurrentTask] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [metrics, setMetrics] = useState({
    tasksCompleted: 0,
    successRate: 0,
    avgExecutionTime: 0,
    activeDevices: 0,
    total_tasks_success: 0,
    total_tasks_failure: 0,
    tasks_in_progress: 0
  })

  const fetchData = useCallback(async () => {
    try {
      setError(null)
      
      // Use API service for all calls
      const [devicesData, tasksData, metricsData] = await Promise.all([
        apiService.getDevices().catch(() => []),
        apiService.getTasks().catch(() => []),
        apiService.getMetrics().catch(() => ({
          total_tasks_success: 0,
          total_tasks_failure: 0,
          tasks_in_progress: 0,
          avg_task_runtime_seconds: 0,
          success_rate: 0
        }))
      ])

      setDevices(devicesData)
      setMetrics({
        tasksCompleted: metricsData.total_tasks_success,
        successRate: metricsData.success_rate,
        avgExecutionTime: metricsData.avg_task_runtime_seconds || 23.4,
        activeDevices: devicesData.length,
        ...metricsData
      })
      setSystemStatus(metricsData.tasks_in_progress > 0 ? 'active' : 'idle')
      
      // Check for running tasks
      const runningTask = tasksData.find(t => t.status === 'running')
      if (runningTask) {
        setCurrentTask({
          description: runningTask.instruction || runningTask.name,
          currentStep: 1,
          totalSteps: 5,
          currentAction: runningTask.instruction || runningTask.name,
          deviceId: runningTask.device_id || 'Unknown',
          activeAgent: 'AutoRL Agent',
          duration: runningTask.duration || '0'
        })
      } else {
        setCurrentTask(null)
      }
    } catch (error) {
      console.error('Failed to fetch data:', error)
      setError('Failed to load dashboard data')
    } finally {
      setLoading(false)
    }
  }, [])

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message) => {
    console.log('📨 Dashboard received WebSocket message:', message)
    
    // Update current task based on WebSocket events
    if (message.event === 'task_started') {
      setCurrentTask({
        description: message.instruction,
        currentStep: 1,
        totalSteps: 5,
        currentAction: 'Starting task...',
        deviceId: message.device_id || 'auto',
        activeAgent: 'AutoRL Agent',
        duration: '0s'
      })
      setSystemStatus('active')
    } else if (message.event === 'perception') {
      setCurrentTask(prev => prev ? {...prev, currentAction: message.text, currentStep: 1} : null)
    } else if (message.event === 'planning') {
      setCurrentTask(prev => prev ? {...prev, currentAction: message.text, currentStep: 2} : null)
    } else if (message.event === 'execution_start') {
      setCurrentTask(prev => prev ? {...prev, currentAction: message.text, currentStep: 3} : null)
    } else if (message.event === 'error') {
      setCurrentTask(prev => prev ? {...prev, currentAction: `⚠️ ${message.text}`, currentStep: 4} : null)
    } else if (message.event === 'recovery_execute') {
      setCurrentTask(prev => prev ? {...prev, currentAction: message.text, currentStep: 4} : null)
    } else if (message.event === 'completed') {
      setCurrentTask(prev => prev ? {...prev, currentAction: '✅ Completed', currentStep: 5} : null)
      // Refresh data after completion
      setTimeout(() => {
        fetchData()
        setCurrentTask(null)
        setSystemStatus('idle')
      }, 2000)
    } else if (message.event === 'task_failed') {
      setCurrentTask(null)
      setSystemStatus('idle')
      fetchData()
    }
  }, [fetchData])

  // WebSocket connection
  const { isConnected } = useWebSocket(handleWebSocketMessage)

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 10000) // Poll less frequently with WebSocket
    return () => clearInterval(interval)
  }, [fetchData])

  if (loading) return <DashboardSkeleton />
  if (error) return <div className="p-6 text-center text-destructive">{error}</div>

  return (
    <div className="container mx-auto p-4 sm:p-6 lg:p-8 max-w-7xl">
      {/* Enhanced Header Section */}
      <div className="mb-8 animate-fade-in">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-3xl sm:text-4xl font-bold mb-3 flex items-center gap-3">
              <div className="relative">
                <Brain className="w-10 h-10 text-primary animate-pulse" />
                <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl animate-pulse" />
              </div>
              <span className="bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                AutoRL AI Agent Platform
              </span>
            </h1>
            <p className="text-muted-foreground text-sm sm:text-base">
              🤖 Autonomous Mobile Automation with AI Agents • {new Date().toLocaleString()}
            </p>
          </div>
          <div className="flex items-center gap-2 sm:gap-3 flex-wrap">
            <Badge 
              variant={isConnected ? 'outline' : 'secondary'} 
              className={`
                text-xs px-3 py-1.5 transition-all duration-300
                ${isConnected 
                  ? 'border-green-500/50 bg-green-500/10 text-green-500 hover:bg-green-500/20' 
                  : 'border-gray-500/50 bg-gray-500/10 text-gray-500'
                }
              `}
            >
              <span className={`
                w-2 h-2 rounded-full mr-1.5 inline-block
                ${isConnected 
                  ? 'bg-green-500 animate-pulse shadow-lg shadow-green-500/50' 
                  : 'bg-gray-400'
                }
              `}></span>
              {isConnected ? 'AI Agent Connected' : 'AI Agent Disconnected'}
            </Badge>
            <Badge 
              variant={systemStatus === 'active' ? 'default' : 'destructive'}
              className={`
                text-xs px-3 py-1.5 transition-all duration-300
                ${systemStatus === 'active' 
                  ? 'bg-primary hover:bg-primary/90 shadow-lg shadow-primary/30' 
                  : ''
                }
              `}
            >
              <Activity className={`w-4 h-4 mr-1.5 ${systemStatus === 'active' ? 'animate-pulse' : ''}`} />
              {systemStatus === 'active' ? 'Agent Active' : 'Agent Idle'}
            </Badge>
          </div>
        </div>
      </div>

      {/* Enhanced Metrics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6 mb-8">
        <div className="animate-fade-in" style={{ animationDelay: '0.1s' }}>
          <MetricCard
            title="Tasks Completed"
            value={metrics.tasksCompleted}
            icon={<CheckCircle className="w-6 h-6" />}
            trend="+12%"
          />
        </div>
        <div className="animate-fade-in" style={{ animationDelay: '0.2s' }}>
          <MetricCard
            title="Success Rate"
            value={`${metrics.successRate.toFixed(1)}%`}
            icon={<Zap className="w-6 h-6" />}
            trend="+2.3%"
          />
        </div>
        <div className="animate-fade-in" style={{ animationDelay: '0.3s' }}>
          <MetricCard
            title="Avg Execution"
            value={`${metrics.avgExecutionTime}s`}
            icon={<Clock className="w-6 h-6" />}
            trend="-5.2s"
          />
        </div>
        <div className="animate-fade-in" style={{ animationDelay: '0.4s' }}>
          <MetricCard
            title="Active Devices"
            value={metrics.activeDevices}
            icon={<Smartphone className="w-6 h-6" />}
            trend="stable"
          />
        </div>
      </div>

      {/* Task Control Panel */}
      <div className="mb-6 animate-fade-in" style={{ animationDelay: '0.5s' }}>
        <TaskControlPanel devices={devices} onTaskCreated={fetchData} />
      </div>

      {/* AI Agent Features Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6 mb-6">
        <Card className="bg-gradient-to-br from-primary/10 to-primary/5 border-primary/20">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 rounded-full bg-primary/20">
                <Brain className="w-6 h-6 text-primary" />
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">AI Agent Status</p>
                <p className="text-2xl font-bold">Intelligent</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              LLM-powered task planning & execution
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-blue-500/10 to-blue-500/5 border-blue-500/20">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 rounded-full bg-blue-500/20">
                <Eye className="w-6 h-6 text-blue-500" />
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">Visual Perception</p>
                <p className="text-2xl font-bold">Active</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              OCR, UI element detection, screenshot analysis
            </p>
          </CardContent>
        </Card>

        <Card className="bg-gradient-to-br from-purple-500/10 to-purple-500/5 border-purple-500/20">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="p-3 rounded-full bg-purple-500/20">
                <Target className="w-6 h-6 text-purple-500" />
              </div>
              <div>
                <p className="text-sm font-medium text-muted-foreground">RL Training</p>
                <p className="text-2xl font-bold">Learning</p>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              Continuous improvement from task execution
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6 mb-6">
        <Card className="lg:col-span-2 transition-all duration-300 hover:shadow-xl hover:shadow-primary/10 border-primary/20">
          <CardHeader className="border-b border-border/50">
            <CardTitle className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-primary/10">
                <Activity className="w-5 h-5 text-primary" />
              </div>
              <span>Live AI Agent Execution</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            {currentTask ? <TaskExecutionView task={currentTask} /> : <EmptyState />}
          </CardContent>
        </Card>

        <Card className="transition-all duration-300 hover:shadow-xl hover:shadow-primary/10 border-primary/20">
          <CardHeader className="border-b border-border/50">
            <CardTitle className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-primary/10">
                <Smartphone className="w-5 h-5 text-primary" />
              </div>
              <span>Connected Devices</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <DeviceStatusPanel devices={devices} />
          </CardContent>
        </Card>
      </div>

      {/* Additional AI Agent Components */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        <div>
          <LiveAgentActivity />
        </div>
        <div>
          <PerformanceCharts />
        </div>
      </div>
    </div>
  )
}

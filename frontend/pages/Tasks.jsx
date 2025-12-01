import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Play, CheckCircle, XCircle, Clock } from 'lucide-react'

export default function Tasks() {
  const [taskDescription, setTaskDescription] = useState('')
  const [recentTasks] = useState([
    {
      id: 1,
      description: 'Open Instagram and like the latest post',
      status: 'completed',
      device: 'Pixel 7',
      duration: '23s',
      timestamp: '2 min ago'
    },
    {
      id: 2,
      description: 'Send a WhatsApp message to John',
      status: 'completed',
      device: 'iPhone 14',
      duration: '15s',
      timestamp: '5 min ago'
    },
    {
      id: 3,
      description: 'Take a screenshot and save to gallery',
      status: 'failed',
      device: 'Pixel 7',
      duration: '8s',
      timestamp: '10 min ago'
    },
    {
      id: 4,
      description: 'Open Settings and enable dark mode',
      status: 'running',
      device: 'iPhone 14',
      duration: '12s',
      timestamp: 'Now'
    }
  ])

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'running':
        return <Clock className="w-4 h-4 text-blue-500 animate-spin" />
      default:
        return null
    }
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Task Execution</h1>
        <p className="text-muted-foreground">
          Create and monitor automation tasks with natural language
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Create New Task</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-2 block">Task Description</label>
              <Textarea
                placeholder="Describe what you want to automate... e.g., 'Open Twitter and post a tweet'"
                value={taskDescription}
                onChange={(e) => setTaskDescription(e.target.value)}
                rows={4}
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-2 block">Target Device</label>
              <Input placeholder="Select device" />
            </div>
            <Button className="w-full">
              <Play className="w-4 h-4 mr-2" />
              Start Task
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Tasks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentTasks.map((task) => (
                <div
                  key={task.id}
                  className="p-4 border rounded-lg hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-start justify-between mb-2">
                    <p className="text-sm font-medium flex-1">{task.description}</p>
                    {getStatusIcon(task.status)}
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{task.device}</span>
                    <span>{task.duration}</span>
                    <span>{task.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

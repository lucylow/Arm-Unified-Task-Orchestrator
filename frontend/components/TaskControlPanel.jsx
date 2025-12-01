import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Play, Loader2 } from 'lucide-react'
import { apiService } from '@/services/api'

export function TaskControlPanel({ devices, onTaskCreated }) {
  const [taskDescription, setTaskDescription] = useState('')
  const [selectedDevice, setSelectedDevice] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!taskDescription.trim()) {
      setError('Please enter a task description')
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      const response = await apiService.createTask(
        taskDescription,
        selectedDevice || null,
        { enable_learning: true }
      )
      
      console.log('Task created:', response)
      
      // Reset form
      setTaskDescription('')
      setSelectedDevice('')
      
      // Notify parent component
      if (onTaskCreated) {
        onTaskCreated()
      }
      
    } catch (error) {
      console.error('Error creating task:', error)
      setError(error.message || 'Failed to create task')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Create New Task</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium mb-2 block">Task Description</label>
            <Textarea
              placeholder="Describe what you want to automate... e.g., 'Open Instagram and like the latest post'"
              value={taskDescription}
              onChange={(e) => setTaskDescription(e.target.value)}
              rows={4}
              disabled={isSubmitting}
              className="resize-none"
            />
          </div>

          <div>
            <label className="text-sm font-medium mb-2 block">Target Device</label>
            <Select value={selectedDevice} onValueChange={setSelectedDevice} disabled={isSubmitting}>
              <SelectTrigger>
                <SelectValue placeholder="Auto-select device" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="">Auto-select</SelectItem>
                {devices && devices.map((device) => (
                  <SelectItem key={device.id} value={device.id}>
                    {device.id} ({device.platform})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {error && (
            <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-md">
              <p className="text-sm text-destructive">{error}</p>
            </div>
          )}

          <Button type="submit" className="w-full" disabled={isSubmitting}>
            {isSubmitting ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Starting Task...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Start Task
              </>
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

export default TaskControlPanel

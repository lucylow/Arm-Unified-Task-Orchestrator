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
    <Card className="border-primary/20 hover:border-primary/40 transition-all duration-300">
      <CardHeader className="border-b border-border/50">
        <CardTitle className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-primary/10">
            <Play className="w-5 h-5 text-primary" />
          </div>
          <span>Create New Task</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-6">
        <form onSubmit={handleSubmit} className="space-y-5">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Task Description
            </label>
            <Textarea
              placeholder="Describe what you want to automate... e.g., 'Open Instagram and like the latest post'"
              value={taskDescription}
              onChange={(e) => setTaskDescription(e.target.value)}
              rows={4}
              disabled={isSubmitting}
              className="resize-none transition-all duration-200 focus:ring-2 focus:ring-primary/20"
            />
            <p className="text-xs text-muted-foreground">
              Be specific about what you want the AI agent to do
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">
              Target Device
            </label>
            <Select value={selectedDevice} onValueChange={setSelectedDevice} disabled={isSubmitting}>
              <SelectTrigger className="transition-all duration-200 focus:ring-2 focus:ring-primary/20">
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
            <div className="p-4 bg-destructive/10 border border-destructive/30 rounded-lg animate-slide-up-fade">
              <p className="text-sm font-medium text-destructive flex items-center gap-2">
                <span className="text-destructive">⚠</span>
                {error}
              </p>
            </div>
          )}

          <Button 
            type="submit" 
            className="w-full transition-all duration-300 hover:shadow-lg hover:shadow-primary/30 hover:-translate-y-0.5" 
            disabled={isSubmitting}
            size="lg"
          >
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

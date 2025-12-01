import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'

export const TaskExecutionView = ({ task }) => (
  <div className="space-y-4">
    <div className="flex items-center justify-between">
      <h3 className="font-semibold text-lg">{task.description}</h3>
      <Badge variant="outline" className="bg-accent">
        Step {task.currentStep}/{task.totalSteps}
      </Badge>
    </div>
    
    <Progress value={(task.currentStep / task.totalSteps) * 100} className="h-2" />
    
    <div className="bg-muted rounded-lg p-4">
      <p className="text-sm text-muted-foreground mb-2">Current Action:</p>
      <p className="font-medium">{task.currentAction}</p>
    </div>
    
    <div className="flex items-center space-x-4 text-sm text-muted-foreground">
      <span>Device: {task.deviceId}</span>
      <span>•</span>
      <span>Agent: {task.activeAgent}</span>
      <span>•</span>
      <span>Duration: {task.duration}s</span>
    </div>
  </div>
)

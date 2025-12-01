import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { CheckCircle2, Clock, Smartphone, Bot, Timer } from 'lucide-react'
import { cn } from '@/lib/utils'

export const TaskExecutionView = ({ task }) => {
  const progress = (task.currentStep / task.totalSteps) * 100
  const isCompleted = task.currentStep === task.totalSteps
  
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header with description and step badge */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-lg mb-1 text-foreground line-clamp-2">
            {task.description}
          </h3>
          <p className="text-sm text-muted-foreground">
            Task execution in progress
          </p>
        </div>
        <Badge 
          variant={isCompleted ? "default" : "outline"} 
          className={cn(
            "shrink-0",
            isCompleted && "bg-green-500 hover:bg-green-600",
            !isCompleted && "bg-primary/10 text-primary border-primary/30"
          )}
        >
          {isCompleted ? (
            <>
              <CheckCircle2 className="w-3 h-3 mr-1" />
              Complete
            </>
          ) : (
            `Step ${task.currentStep}/${task.totalSteps}`
          )}
        </Badge>
      </div>
      
      {/* Enhanced Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Progress</span>
          <span className="font-semibold text-foreground">{Math.round(progress)}%</span>
        </div>
        <div className="relative">
          <Progress 
            value={progress} 
            className={cn(
              "h-3 transition-all duration-500",
              isCompleted && "bg-green-500"
            )} 
          />
          <div 
            className={cn(
              "absolute top-0 left-0 h-3 rounded-full transition-all duration-500",
              "bg-gradient-to-r from-primary via-primary/90 to-primary",
              isCompleted && "from-green-500 via-green-400 to-green-500",
              "shadow-lg shadow-primary/50"
            )}
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>
      
      {/* Current Action Card */}
      <div className={cn(
        "relative overflow-hidden rounded-xl p-5",
        "bg-gradient-to-br from-primary/10 via-primary/5 to-transparent",
        "border border-primary/20",
        "transition-all duration-300",
        "hover:border-primary/40 hover:shadow-lg hover:shadow-primary/10"
      )}>
        <div className="flex items-start gap-3">
          <div className={cn(
            "flex-shrink-0 p-2 rounded-lg",
            "bg-primary/20 text-primary",
            "animate-pulse"
          )}>
            <Clock className="w-4 h-4" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-muted-foreground mb-1.5 uppercase tracking-wide">
              Current Action
            </p>
            <p className="font-semibold text-foreground leading-relaxed">
              {task.currentAction}
            </p>
          </div>
        </div>
        {/* Animated shimmer effect */}
        <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/5 to-transparent pointer-events-none" />
      </div>
      
      {/* Task Metadata */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
        <div className="flex items-center gap-2 text-sm">
          <div className="p-1.5 rounded-md bg-muted">
            <Smartphone className="w-3.5 h-3.5 text-muted-foreground" />
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground">Device</p>
            <p className="font-medium text-foreground truncate">{task.deviceId}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2 text-sm">
          <div className="p-1.5 rounded-md bg-muted">
            <Bot className="w-3.5 h-3.5 text-muted-foreground" />
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground">Agent</p>
            <p className="font-medium text-foreground truncate">{task.activeAgent}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2 text-sm">
          <div className="p-1.5 rounded-md bg-muted">
            <Timer className="w-3.5 h-3.5 text-muted-foreground" />
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground">Duration</p>
            <p className="font-medium text-foreground">{task.duration}s</p>
          </div>
        </div>
      </div>
    </div>
  )
}
